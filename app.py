#!/usr/bin/env python3
"""
Full Thesis Single-file Flask app for IOT_Chicken_Monitoring.
- Assumes ML libs installed (YOLOv8, OpenCV, TensorFlow).
- Designed to run with XAMPP/phpMyAdmin MySQL (DB name default from env).
- All logic in one file: auth, DB, ingestion, fusion background, vision stream.
"""

import os
import sys
import json
import time
import random
import string
import logging
import threading
import atexit
from pathlib import Path
from datetime import datetime, timedelta, date
from functools import wraps

from flask import (
    Flask, request, jsonify, render_template, redirect, url_for,
    session, flash, send_from_directory, Response, abort
)

# DB: mysql.connector pooling
import mysql.connector
from mysql.connector import pooling

# Security
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ML / audio / cv libs (assumed installed)
import numpy as np
from ultralytics import YOLO
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# -------------------------
# Logging
# -------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger("chicken_monitor")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# -------------------------
# Folders & config
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "uploads"
AUDIO_DIR = UPLOAD_DIR / "audio"
IMG_DIR = UPLOAD_DIR / "images"
TMP_DIR = UPLOAD_DIR / "tmp"

for p in (UPLOAD_DIR, AUDIO_DIR, IMG_DIR, TMP_DIR, MODEL_DIR):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        logger.exception("Failed to create folder %s", p)

ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg"}
ALLOWED_AUDIO_EXT = {"wav", "mp3", "m4a", "flac"}

# thresholds (tweak as necessary)
ENV_MSE_THRESHOLD = float(os.environ.get("ENV_MSE_THRESHOLD", 800.0))
AUDIO_MSE_THRESHOLD = float(os.environ.get("AUDIO_MSE_THRESHOLD", 0.6))
VISION_MSE_THRESHOLD = float(os.environ.get("VISION_MSE_THRESHOLD", 250.0))
FUSION_HIGH_CONF = float(os.environ.get("FUSION_HIGH_CONF", 0.8))

# Alert Cooldowns
HUMIDITY_ALERT_COOLDOWN = int(os.environ.get("HUMIDITY_ALERT_COOLDOWN", 1800)) # 30 mins
last_humidity_alert = 0

# DB (XAMPP defaults or env)
DB_HOST = os.environ.get("MYSQL_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("MYSQL_PORT", 3306))
DB_USER = os.environ.get("MYSQL_USER", "root")
DB_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")  # default XAMPP blank
DB_NAME = os.environ.get("MYSQL_DB", "research")

POOL_NAME = "research_pool"
POOL_SIZE = int(os.environ.get("MYSQL_POOL_SIZE", 5))

dbconfig = {
    "host": DB_HOST,
    "port": DB_PORT,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "database": DB_NAME,
    "autocommit": False
}

connection_pool = None
try:
    connection_pool = pooling.MySQLConnectionPool(pool_name=POOL_NAME, pool_size=POOL_SIZE, **dbconfig)
    logger.info("DB pool created: %s (size=%s)", POOL_NAME, POOL_SIZE)
except Exception:
    logger.exception("Failed to create DB pool — will fallback to direct connect")

# -------------------------
# DB connection fix
# -------------------------
def db_connect():
    """Safely obtain a connection from the pool, with auto-recovery."""
    global connection_pool
    try:
        if connection_pool:
            try:
                return connection_pool.get_connection()
            except mysql.connector.errors.PoolError:
                logger.warning("⚠️ Connection pool exhausted — recreating pool")
                connection_pool = pooling.MySQLConnectionPool(
                    pool_name=POOL_NAME,
                    pool_size=POOL_SIZE,
                    **dbconfig
                )
                return connection_pool.get_connection()
        else:
            return mysql.connector.connect(**dbconfig)
    except Exception as e:
        logger.exception("db_connect failed: %s", e)
        raise


def _get_conn_cursor(dictionary=True):
    """Always returns a (conn, cursor) safely, ensuring cleanup responsibility."""
    conn = db_connect()
    cursor = conn.cursor(dictionary=dictionary, buffered=True)
    return conn, cursor


# -------------------------
# Flask app
# -------------------------
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))
app.secret_key = os.environ.get("FLASK_SECRET", os.urandom(24))
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
app.config["DEBUG"] = bool(int(os.environ.get("DEBUG", "0")))

# -------------------------
# Load models (non-lazy for this full version)
# -------------------------
yolo_model = None
cnn_model = None
lstm_model = None
audio_model = None
fusion_model = None

def _load_yolo():
    global yolo_model
    try:
        path = MODEL_DIR / "best.pt"
        if path.exists():
            yolo_model = YOLO(str(path))
            logger.info("Loaded YOLO from %s", path)
        else:
            logger.warning("YOLO model not found at %s", path)
    except Exception:
        logger.exception("Failed to load YOLO")
        yolo_model = None

def _load_tf_models():
    global cnn_model, lstm_model, audio_model
    try:
        cnn_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        logger.info("Loaded ResNet50 feature extractor")
    except Exception:
        logger.exception("Failed to load ResNet50")
        cnn_model = None
    try:
        f = MODEL_DIR / "chicken_behavior_lstm.h5"
        if f.exists():
            lstm_model = load_model(str(f))
            logger.info("Loaded LSTM model %s", f)
        else:
            logger.warning("LSTM model not present: %s", f)
    except Exception:
        logger.exception("Failed to load LSTM")
        lstm_model = None
    try:
        f = MODEL_DIR / "crnn_autoencoder.h5"
        if f.exists():
            audio_model = load_model(str(f), compile=False)
            logger.info("Loaded audio autoencoder %s", f)
        else:
            logger.warning("Audio model not present: %s", f)
    except Exception:
        logger.exception("Failed to load audio model")
        audio_model = None

def _load_fusion_model():
    global fusion_model
    try:
        p = MODEL_DIR / "fusion_model.pkl"
        if p.exists():
            import pickle
            with open(p, "rb") as fh:
                fusion_model = pickle.load(fh)
            logger.info("Loaded fusion_model.pkl")
        else:
            logger.warning("fusion_model.pkl not found at %s", p)
    except Exception:
        logger.exception("Failed to load fusion_model.pkl")
        fusion_model = None

# -------------------------
# Utilities
# -------------------------
def allowed_file(filename, allowed_set):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_set

def rand_suffix(n=8):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))

def generate_tracked_id():
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    return f"C{ts[-10:]}{rand_suffix(3)}"

def save_uploaded_file(file_storage, dest_folder: Path):
    fn = secure_filename(file_storage.filename or "upload")
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    name = f"{ts}_{rand_suffix(6)}_{fn}"
    path = dest_folder / name
    file_storage.save(str(path))
    return str(path)

# simple environmental mse calculator (example heuristic)
def compute_env_mse(temp, hum, gas):
    try:
        # compare to comfortable baseline (26C, 60% hum, 300ppm)
        t = float(temp or 0); h = float(hum or 0); g = float(gas or 0)
        mse = ((t - 26.0)**2 + (h - 60.0)**2 + ((g - 300.0)/10.0)**2) / 3.0
        return float(mse)
    except Exception:
        return 0.0

# -------------------------
# DB helper
# -------------------------
def _get_conn_cursor(dictionary=True):
    conn = db_connect()
    cursor = conn.cursor(dictionary=dictionary, buffered=True)
    return conn, cursor

# -------------------------
# CRUD functions (use dictionary param consistently)
# -------------------------
def create_user(username, email, password_plain, farm_name=None, location=None, role="user"):
    pw_hash = generate_password_hash(password_plain)
    conn, cur = _get_conn_cursor(dictionary=False)
    try:
        cur.execute("""
            INSERT INTO users (username, email, password_hash, farm_name, location, role, created_at)
            VALUES (%s,%s,%s,%s,%s,%s,NOW())
        """, (username, email, pw_hash, farm_name, location, role))
        conn.commit()
        return cur.lastrowid
    finally:
        cur.close(); conn.close()

def get_user_by_username_or_email(ident):
    conn, cur = _get_conn_cursor(dictionary=True)
    try:
        cur.execute("SELECT * FROM users WHERE username=%s OR email=%s LIMIT 1", (ident, ident))
        return cur.fetchone()
    finally:
        cur.close(); conn.close()

def get_user_by_id(user_id):
    conn, cur = _get_conn_cursor(dictionary=True)
    try:
        cur.execute("SELECT id, username, email, farm_name, location, role, created_at FROM users WHERE id=%s LIMIT 1", (int(user_id),))
        return cur.fetchone()
    finally:
        cur.close(); conn.close()

# Environmental
def insert_environmental_data(user_id, temperature, humidity, gas, mse_score=0.0, is_anomaly=0, timestamp=None, device_id=None):
    conn, cur = _get_conn_cursor(dictionary=False)
    ts = timestamp or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("""
        INSERT INTO environmental_data (user_id, device_id, timestamp, temperature, humidity, gas, mse_score, is_anomaly)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    """, (int(user_id), device_id, ts, float(temperature), float(humidity), float(gas), float(mse_score), int(bool(is_anomaly))))
    conn.commit()
    rowid = cur.lastrowid
    cur.close(); conn.close()
    return rowid

def fetch_environmental_history(user_id, limit=200):
    conn, cur = _get_conn_cursor(dictionary=True)
    try:
        cur.execute("SELECT * FROM environmental_data WHERE user_id=%s ORDER BY timestamp DESC LIMIT %s", (int(user_id), int(limit)))
        return cur.fetchall()
    finally:
        cur.close(); conn.close()

# Audio
def insert_audio_data(user_id, rms, peak, audio_db, file_path=None, mse_score=0.0, is_anomaly=0, timestamp=None, device_id=None):
    conn, cur = _get_conn_cursor(dictionary=False)
    ts = timestamp or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("""
        INSERT INTO audio_data (user_id, device_id, timestamp, rms, peak, audio_db, file_path, mse_score, is_anomaly)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (int(user_id), device_id, ts, float(rms), float(peak), float(audio_db), file_path, float(mse_score), int(bool(is_anomaly))))
    conn.commit()
    rid = cur.lastrowid
    cur.close(); conn.close()
    return rid

def fetch_audio_history(user_id, limit=200):
    conn, cur = _get_conn_cursor(dictionary=True)
    try:
        cur.execute("SELECT * FROM audio_data WHERE user_id=%s ORDER BY timestamp DESC LIMIT %s", (int(user_id), int(limit)))
        return cur.fetchall()
    finally:
        cur.close(); conn.close()

# Vision
def insert_vision_data(user_id, behavior, bbox, mse_score=0.0, is_anomaly=0, timestamp=None, frame_path=None, device_id=None):
    conn, cur = _get_conn_cursor(dictionary=False)
    ts = timestamp or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("""
        INSERT INTO vision_data (user_id, device_id, timestamp, behavior, bbox, file_path, mse_score, is_anomaly)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    """, (int(user_id), device_id, ts, behavior, bbox, frame_path, float(mse_score), int(bool(is_anomaly))))
    conn.commit()
    vid = cur.lastrowid
    cur.close(); conn.close()
    return vid

def fetch_vision_history(user_id, limit=200):
    conn, cur = _get_conn_cursor(dictionary=True)
    try:
        cur.execute("SELECT * FROM vision_data WHERE user_id=%s ORDER BY timestamp DESC LIMIT %s", (int(user_id), int(limit)))
        return cur.fetchall()
    finally:
        cur.close(); conn.close()

# Fusion
def insert_fusion_prediction(user_id, sensor_anom, audio_anom, vision_anom,
                             sensor_score, audio_score, vision_score,
                             fusion_label, fusion_confidence, timestamp=None):
    conn, cur = _get_conn_cursor(dictionary=False)
    ts = timestamp or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("""
        INSERT INTO fusion_predictions
        (user_id, timestamp, sensor_anomaly, audio_anomaly, vision_anomaly,
         sensor_score, audio_score, vision_score, fusion_label, fusion_confidence)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (int(user_id), ts, int(bool(sensor_anom)), int(bool(audio_anom)), int(bool(vision_anom)),
          float(sensor_score), float(audio_score), float(vision_score), fusion_label, float(fusion_confidence)))
    conn.commit()
    fid = cur.lastrowid
    cur.close(); conn.close()
    return fid

def fetch_fusion_history(user_id, limit=200):
    conn, cur = _get_conn_cursor(dictionary=True)
    try:
        cur.execute("SELECT * FROM fusion_predictions WHERE user_id=%s ORDER BY timestamp DESC LIMIT %s", (int(user_id), int(limit)))
        return cur.fetchall()
    finally:
        cur.close(); conn.close()

# Tracked & alerts & eggs & devices
def insert_tracked_chicken(user_id, tracked_id, frame_path, vision_bbox, mse_score, related_fusion_id=None):
    conn, cur = _get_conn_cursor(dictionary=False)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("""
        INSERT INTO tracked_chickens (user_id, tracked_id, frame_path, vision_bbox, mse_score, first_seen, last_seen, recovered, related_fusion_id)
        VALUES (%s,%s,%s,%s,%s,%s,%s,0,%s)
    """, (int(user_id), tracked_id, frame_path, vision_bbox, float(mse_score), now, now, related_fusion_id))
    conn.commit()
    tid = cur.lastrowid
    cur.close(); conn.close()
    return tid

def update_tracked_chicken_lastseen(tracked_id, user_id, frame_path=None, mse_score=None, related_fusion_id=None):
    conn, cur = _get_conn_cursor(dictionary=False)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    sets = ["last_seen=%s"]; args = [now]
    if frame_path:
        sets.append("frame_path=%s"); args.append(frame_path)
    if mse_score is not None:
        sets.append("mse_score=%s"); args.append(float(mse_score))
    if related_fusion_id is not None:
        sets.append("related_fusion_id=%s"); args.append(int(related_fusion_id))
    args.extend([tracked_id, int(user_id)])
    sql = "UPDATE tracked_chickens SET " + ",".join(sets) + " WHERE tracked_id=%s AND user_id=%s"
    cur.execute(sql, tuple(args))
    conn.commit()
    rc = cur.rowcount
    cur.close(); conn.close()
    return rc

def mark_tracked_chicken_recovered(tracked_id, user_id):
    conn, cur = _get_conn_cursor(dictionary=False)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("""UPDATE tracked_chickens SET recovered=1, recovery_timestamp=%s, last_seen=%s WHERE tracked_id=%s AND user_id=%s""",
                (now, now, tracked_id, int(user_id)))
    conn.commit()
    rc = cur.rowcount
    cur.close(); conn.close()
    return rc

def fetch_tracked_for_user(user_id, limit=200):
    conn, cur = _get_conn_cursor(dictionary=True)
    try:
        cur.execute("SELECT * FROM tracked_chickens WHERE user_id=%s ORDER BY last_seen DESC LIMIT %s", (int(user_id), int(limit)))
        return cur.fetchall()
    finally:
        cur.close(); conn.close()

def create_alert(user_id, type_, message, severity='info'):
    conn, cur = _get_conn_cursor(dictionary=False)
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("INSERT INTO alerts (user_id, type, message, severity, timestamp) VALUES (%s,%s,%s,%s,%s)",
                (int(user_id), type_, message, severity, ts))
    conn.commit()
    aid = cur.lastrowid
    cur.close(); conn.close()
    return aid

def fetch_alerts(user_id, limit=200):
    conn, cur = _get_conn_cursor(dictionary=True)
    try:
        cur.execute("SELECT * FROM alerts WHERE user_id=%s ORDER BY timestamp DESC LIMIT %s", (int(user_id), int(limit)))
        return cur.fetchall()
    finally:
        cur.close(); conn.close()

def resolve_alert(alert_id, user_id):
    conn, cur = _get_conn_cursor(dictionary=False)
    cur.execute("UPDATE alerts SET resolved=1 WHERE id=%s AND user_id=%s", (int(alert_id), int(user_id)))
    conn.commit()
    rc = cur.rowcount
    cur.close(); conn.close()
    return rc

def insert_egg_production(user_id, date_value, total_eggs, broken_eggs, avg_stress_label=None, notes=None):
    conn, cur = _get_conn_cursor(dictionary=False)
    cur.execute("INSERT INTO egg_production (user_id, date, total_eggs, broken_eggs, avg_stress_label, notes, created_at) VALUES (%s,%s,%s,%s,%s,%s,NOW())",
                (int(user_id), date_value, int(total_eggs), int(broken_eggs), avg_stress_label, notes))
    conn.commit()
    eid = cur.lastrowid
    cur.close(); conn.close()
    return eid

def fetch_egg_production(user_id, limit=200):
    conn, cur = _get_conn_cursor(dictionary=True)
    try:
        cur.execute("SELECT * FROM egg_production WHERE user_id=%s ORDER BY date DESC LIMIT %s", (int(user_id), int(limit)))
        return cur.fetchall()
    finally:
        cur.close(); conn.close()

def insert_device(user_id, device_uid, device_type=None, location=None, status='offline'):
    conn, cur = _get_conn_cursor(dictionary=False)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cur.execute("INSERT INTO devices (user_id, device_uid, device_type, location, status, last_seen, created_at) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (int(user_id), device_uid, device_type, location, status, now, now))
        conn.commit()
    except mysql.connector.IntegrityError:
        conn.rollback()
        cur.execute("UPDATE devices SET user_id=%s, device_type=%s, location=%s, status=%s, last_seen=%s WHERE device_uid=%s",
                    (int(user_id), device_type, location, status, now, device_uid))
        conn.commit()
    finally:
        rowid = cur.lastrowid
        cur.close(); conn.close()
        return rowid

def update_device_last_seen(device_uid, status='connected'):
    conn, cur = _get_conn_cursor(dictionary=False)
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("UPDATE devices SET last_seen=%s, status=%s WHERE device_uid=%s", (now, status, device_uid))
    conn.commit()
    rc = cur.rowcount
    cur.close(); conn.close()
    return rc

def fetch_devices_for_user(user_id):
    conn, cur = _get_conn_cursor(dictionary=True)
    try:
        cur.execute("SELECT * FROM devices WHERE user_id=%s ORDER BY last_seen DESC", (int(user_id),))
        return cur.fetchall()
    finally:
        cur.close(); conn.close()

# -------------------------
# Convenience: latest rows for fusion
# -------------------------
def fetch_latest_rows_all():
    conn, cur = _get_conn_cursor(dictionary=True)
    try:
        cur.execute("SELECT id, user_id, mse_score as sensor_score, is_anomaly as sensor_flag, timestamp FROM environmental_data ORDER BY id DESC LIMIT 1")
        sensor_row = cur.fetchone()
        cur.execute("SELECT id, user_id, mse_score as audio_score, is_anomaly as audio_flag, timestamp FROM audio_data ORDER BY id DESC LIMIT 1")
        audio_row = cur.fetchone()
        cur.execute("SELECT id, user_id, mse_score as vision_score, is_anomaly as vision_flag, timestamp FROM vision_data ORDER BY id DESC LIMIT 1")
        vision_row = cur.fetchone()
        return sensor_row, audio_row, vision_row
    finally:
        cur.close(); conn.close()

logger.info("DB helpers ready")

# -------------------------
# Fusion & Tracking logic
# -------------------------
active_tracks = {}
FUSION_INTERVAL = int(os.environ.get("FUSION_INTERVAL", 10))
FUSION_TIME_WINDOW = int(os.environ.get("FUSION_TIME_WINDOW", 60))
TRACK_RECOVERY_TIMEOUT = int(os.environ.get("TRACK_RECOVERY_TIMEOUT", 300))

def call_fusion(sensor_score, audio_score, vision_score, sensor_flag, audio_flag, vision_flag):
    try:
        # STRICT RULE-BASED LOGIC (User Requested)
        # 1. Vision Only -> "Anomaly (Behavior)"
        # 2. Vision + (Sensor OR Audio) -> "Mild Stress"
        # 3. Vision + Sensor + Audio -> "High Stress"
        # 4. Sensor Only -> "Sensor Alert"
        # 5. Audio Only -> "Audio Alert"
        
        s = int(bool(sensor_flag))
        a = int(bool(audio_flag))
        v = int(bool(vision_flag))
        n = s + a + v

        # Default confidence
        conf = 0.95

        if v:
            if s and a:
                return "High Stress", 0.98
            elif s or a:
                return "Mild Stress", 0.90
            else:
                return "Anomaly (Behavior)", 0.85
        
        if s: return "Sensor Alert", 0.85
        if a: return "Audio Alert", 0.85
        
        # If no flags, check model score (optional fallback, but flags take precedence)
        if fusion_model is not None:
            try:
                x = np.array([[sensor_score, audio_score, vision_score]], dtype=float)
                pred = fusion_model.predict(x)
                if isinstance(pred, np.ndarray) or isinstance(pred, list):
                    model_conf = float(np.max(pred)) if hasattr(pred, '__len__') else float(pred)
                    if model_conf > 0.6: # Only if model is somewhat confident
                         return "Stress", model_conf
            except: pass

        return "Normal", 0.95

    except Exception:
        logger.exception("call_fusion error")
        return "Unknown", 0.0

def handle_binary_tracking(user_id, sensor_flag, audio_flag, vision_flag, frame_path=None, vision_bbox=None, mse_score=None, fusion_confidence=None, fusion_id=None):
    try:
        user_id = int(user_id or 1)
    except Exception:
        user_id = 1
    flags = (int(bool(sensor_flag)), int(bool(audio_flag)), int(bool(vision_flag)))
    user_tracks = active_tracks.setdefault(user_id, {})
    if flags == (1,1,1):
        tracked_id = generate_tracked_id()
        try:
            db_row_id = insert_tracked_chicken(user_id, tracked_id, frame_path or "", json.dumps({"bbox": vision_bbox} if vision_bbox else {}), mse_score or 0.0, related_fusion_id=fusion_id)
            user_tracks[tracked_id] = {"db_id": db_row_id, "first_seen": datetime.utcnow(), "last_seen": datetime.utcnow(), "frame_path": frame_path, "vision_bbox": vision_bbox, "mse": mse_score, "fusion_confidence": fusion_confidence}
        except Exception:
            logger.exception("Failed to insert tracked chicken")
        try:
            create_alert(user_id, "Fusion", f"New stressed chicken detected: {tracked_id}", "danger")
        except Exception:
            logger.exception("Failed to create alert")
        logger.info("New tracked_id=%s for user=%s", tracked_id, user_id)
        return tracked_id
    if flags == (0,0,0):
        recovered = []
        for tid in list(user_tracks.keys()):
            try:
                mark_tracked_chicken_recovered(tid, user_id)
                recovered.append(tid)
                user_tracks.pop(tid, None)
                create_alert(user_id, "Fusion", f"Tracked chicken {tid} marked recovered", "info")
            except Exception:
                logger.exception("Error marking recovered")
        return recovered if recovered else None
    if user_tracks:
        try:
            recent_tid = max(user_tracks.items(), key=lambda p: p[1].get("last_seen"))[0]
            update_tracked_chicken_lastseen(recent_tid, user_id, frame_path=frame_path, mse_score=mse_score, related_fusion_id=fusion_id)
            user_tracks[recent_tid]["last_seen"] = datetime.utcnow()
            if frame_path: user_tracks[recent_tid]["frame_path"] = frame_path
            return recent_tid
        except Exception:
            logger.exception("Failed to update last_seen")
            return None
    return None

def cleanup_stale_tracks():
    now = datetime.utcnow()
    for user_id, tracks in list(active_tracks.items()):
        for tid, meta in list(tracks.items()):
            last_seen = meta.get("last_seen") or meta.get("first_seen")
            if last_seen and (now - last_seen).total_seconds() > TRACK_RECOVERY_TIMEOUT:
                try:
                    mark_tracked_chicken_recovered(tid, user_id)
                except Exception:
                    logger.exception("Error recovering stale track %s", tid)
                tracks.pop(tid, None)
                try:
                    create_alert(user_id, "Fusion", f"Tracked chicken {tid} auto-marked recovered (timeout)", "info")
                except Exception:
                    logger.exception("Failed to create alert")

def run_fusion_once():
    try:
        sensor_row, audio_row, vision_row = fetch_latest_rows_all()
        # --------------------------------------------------
        # FORCE audio anomaly to follow audio_data.is_anomaly
        # --------------------------------------------------
        if audio_row:
            try:
                audio_flag = int(audio_row.get("audio_flag") or audio_row.get("is_anomaly", 0))
            except:
                audio_flag = 0
        else:
            audio_flag = 0


        # =====================================
        # FORCE sensor anomaly = 0 if environmental_data was normal
        # =====================================
        if sensor_row:
            try:
                env_flag = int(sensor_row.get("is_anomaly", 0))
                sensor_flag_override = 1 if env_flag == 1 else 0
            except:
                sensor_flag_override = 0
        else:
            sensor_flag_override = 0

        # Filter out stale rows
        now_dt = datetime.utcnow()
        valid_rows = []
        for r in [sensor_row, audio_row, vision_row]:
            if not r:
                valid_rows.append(None)
                continue
            t = r.get("timestamp")
            if isinstance(t, str):
                try: t = datetime.fromisoformat(t)
                except: 
                    try: t = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
                    except: t = None
            
            # If data is older than window + buffer (e.g. 2 mins), ignore it
            if t and (now_dt - t).total_seconds() > (FUSION_TIME_WINDOW * 2):
                valid_rows.append(None)
            else:
                valid_rows.append(r)
        
        sensor_row, audio_row, vision_row = valid_rows

        if not any((sensor_row, audio_row, vision_row)): return None
        user_id = (sensor_row or audio_row or vision_row).get("user_id", 1)
        def _get_vals(row, score_key, flag_key):
            if not row: return 0.0, 0
            return float(row.get(score_key) or 0.0), int(bool(row.get(flag_key)))
        sensor_score = float(sensor_row.get("mse_score", 0.0)) if sensor_row else 0.0
        sensor_flag = sensor_flag_override if sensor_row else 0
        audio_score = float(audio_row.get("audio_score", 0.0)) if audio_row else 0.0
        vision_score, vision_flag = _get_vals(vision_row, "vision_score", "vision_flag")
        # time-window check (only for valid concurrent data)
        times = []
        for r in (sensor_row, audio_row, vision_row):
            if not r: continue
            t = r.get("timestamp")
            if not t: continue
            if isinstance(t, str):
                try: t = datetime.fromisoformat(t)
                except Exception:
                    try: t = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
                    except Exception: t = None
            if t: times.append(t)
        if times and len(times) > 1:
            if (max(times) - min(times)).total_seconds() > FUSION_TIME_WINDOW:
                logger.debug("Skipping fusion due to time window mismatch")
                return None
        label, conf = call_fusion(sensor_score, audio_score, vision_score, sensor_flag, audio_flag, vision_flag)
        fid = insert_fusion_prediction(user_id, sensor_flag, audio_flag, vision_flag, sensor_score, audio_score, vision_score, label, conf)
        tracked_result = handle_binary_tracking(user_id, sensor_flag, audio_flag, vision_flag, frame_path=None, vision_bbox=None, mse_score=vision_score, fusion_confidence=conf, fusion_id=fid)
        if label != "Normal" and label != "Unknown" and conf >= FUSION_HIGH_CONF:
            severity = "danger" if "High" in label or "Multi" in label else "warning"
            try: create_alert(user_id, "Fusion", f"{label} detected (conf={conf:.2f})", severity)
            except Exception: logger.exception("Failed to create alert")
        logger.info("Fusion inserted id=%s label=%s conf=%.2f tracked=%s", fid, label, float(conf), tracked_result)
        return {"fusion_id": fid, "label": label, "confidence": conf, "tracked": tracked_result}
    except Exception:
        logger.exception("run_fusion_once failed")
        return None

def fusion_background_loop(interval_seconds=FUSION_INTERVAL):
    logger.info("Starting fusion background loop")
    while True:
        try:
            run_fusion_once()
            cleanup_stale_tracks()
        except Exception:
            logger.exception("Exception in fusion loop")
        time.sleep(interval_seconds)

# start fusion thread
fusion_thread = threading.Thread(target=fusion_background_loop, args=(FUSION_INTERVAL,), daemon=True)
fusion_thread.start()
logger.info("Fusion background thread started")

# -------------------------
# Authentication & session
# -------------------------
SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", 60 * 60 * 4))

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        uid = session.get("user_id")
        if not uid:
            return redirect(url_for("login"))
        last = session.get("_last_active")
        if last:
            try:
                last_dt = datetime.fromisoformat(last)
                if (datetime.utcnow() - last_dt).total_seconds() > SESSION_TIMEOUT:
                    session.clear()
                    flash("Session expired", "warning")
                    return redirect(url_for("login"))
            except Exception:
                session.clear()
                return redirect(url_for("login"))
        session["_last_active"] = datetime.utcnow().isoformat()
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

def admin_required(f):
    @wraps(f)
    def inner(*args, **kwargs):
        uid = session.get("user_id")
        if not uid:
            return redirect(url_for("login"))
        user = get_user_by_id(uid)
        if not user or user.get("role") != "admin":
            flash("Admin access required", "danger")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    inner.__name__ = f.__name__
    return inner

# -------------------------
# Routes: auth
# -------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        farm_name = request.form.get("farm_name", "")
        location = request.form.get("location", "")
        if not username or not email or not password:
            flash("username, email and password are required", "warning")
            return redirect(url_for("signup"))
        try:
            uid = create_user(username, email, password, farm_name or None, location or None, role="user")
            flash("Account created, please login", "success")
            return redirect(url_for("login"))
        except Exception:
            flash("Failed to create account", "danger")
            return redirect(url_for("signup"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        ident = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not ident or not password:
            flash("Username and password required", "warning")
            return redirect(url_for("login"))
        user = get_user_by_username_or_email(ident)
        if not user:
            flash("Invalid credentials", "danger")
            return redirect(url_for("login"))
        stored_pw = user.get("password_hash", "")
        try:
            if stored_pw == password or check_password_hash(stored_pw, password):
                session.clear()
                session["user_id"] = int(user["id"])
                session["username"] = user.get("username")
                session["farm_name"] = user.get("farm_name") or "Farm"
                session["_last_active"] = datetime.utcnow().isoformat()
                flash("Login successful", "success")
                return redirect(url_for("dashboard"))
            else:
                flash("Invalid credentials", "danger")
                return redirect(url_for("login"))
        except Exception:
            logger.exception("Password verification failed")
            flash("Login error", "danger")
            return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out", "info")
    return redirect(url_for("login"))

# -------------------------
# Dashboard & pages
# -------------------------
@app.route("/")
@login_required
def dashboard():
    user_id = session.get("user_id", 1)
    try:
        latest_env = fetch_environmental_history(user_id, limit=1)
        latest_env = latest_env[0] if latest_env else None
        fusions = fetch_fusion_history(user_id, limit=10)
        alerts = fetch_alerts(user_id, limit=10)
    except Exception:
        logger.exception("Dashboard data fetch failed")
        latest_env, fusions, alerts = None, [], []
    return render_template("dashboard.html", latest=latest_env, fusions=fusions, alerts=alerts)

@app.route("/devices")
@login_required
def devices_page():
    user_id = session.get("user_id", 1)
    devices = fetch_devices_for_user(user_id)
    return render_template("devices.html", devices=devices)

@app.route("/environment")
@login_required
def environment_page():
    user_id = session.get("user_id", 1)
    data = fetch_environmental_history(user_id, limit=300)
    return render_template("environment.html", data=data)

@app.route("/audio")
@login_required
def audio_page():
    user_id = session.get("user_id", 1)
    data = fetch_audio_history(user_id, limit=300)
    return render_template("audio.html", data=data)

@app.route("/vision")
@login_required
def vision_page():
    user_id = session.get("user_id", 1)
    data = fetch_vision_history(user_id, limit=300)
    return render_template("vision.html", data=data)

@app.route("/analytics")
@login_required
def analytics_page():
    user_id = session.get("user_id", 1)
    data = fetch_fusion_history(user_id, limit=500)
    return render_template("analytics.html", data=data)

@app.route("/eggs")
@login_required
def eggs_page():
    user_id = session.get("user_id", 1)
    rows = fetch_egg_production(user_id, limit=200)
    return render_template("eggs.html", data=rows)

@app.route("/history")
@login_required
def history_page():
    return render_template("history.html")






# -------------------------
# Public APIs for devices & uploads
# -------------------------
@app.route("/api/devices/register", methods=["POST"])
@login_required
def api_register_device():
    user_id = session.get("user_id", 1)
    device_uid = request.form.get("device_uid") or (request.json and request.json.get("device_uid"))
    device_type = request.form.get("device_type") or (request.json and request.json.get("device_type"))
    location = request.form.get("location") or (request.json and request.json.get("location"))
    if not device_uid:
        return jsonify({"error":"device_uid required"}), 400
    try:
        did = insert_device(user_id, device_uid, device_type, location, status="connected")
        return jsonify({"status":"ok","device_id":did})
    except Exception:
        logger.exception("api_register_device failed")
        return jsonify({"error":"failed"}), 500

@app.route("/api/devices/heartbeat", methods=["POST"])
def api_device_heartbeat():
    data = request.get_json(silent=True) or request.form
    device_uid = data.get("device_uid")
    status = data.get("status", "connected")
    if not device_uid:
        return jsonify({"error":"device_uid required"}), 400
    try:
        rows = update_device_last_seen(device_uid, status=status)
        return jsonify({"status":"ok","rows":rows})
    except Exception:
        logger.exception("api_device_heartbeat failed")
        return jsonify({"error":"failed"}), 500

# -------------------------
# Sensor ingestion (ESP32) - ensures file_path exists
# -------------------------
@app.route("/api/sensor", methods=["POST"])
def api_sensor():
    """Receive JSON data from ESP32 and insert into MySQL with MSE and anomaly flags."""
    try:
        data = request.get_json(force=True)
        if not data:
            logger.error("api_sensor: No JSON received")
            return jsonify({"error": "no JSON received"}), 400

        logger.info(f"api_sensor received data: {data}")

        user_id = int(data.get("user_id", 1))
        device_id = data.get("device_id", "esp32_001")

        temperature = float(data.get("temperature", 0))
        humidity = float(data.get("humidity", 0))
        gas = float(data.get("gas", 0))
        mic_rms = float(data.get("mic_rms", 0))
        mic_peak = float(data.get("mic_peak", 0))
        audio_db = float(data.get("audio_level_db", 0))

        # --- Compute error scores ---
        # ==============================
        # ENVIRONMENTAL ANOMALY CHECK
        # ==============================

        global last_humidity_alert

        # MSE still logged for analytics
        env_mse = compute_env_mse(temperature, humidity, gas)

        # Model-Based Anomaly Detection
        if env_mse > ENV_MSE_THRESHOLD:
            env_anomaly = 1
        else:
            env_anomaly = 0
            
        # Fallback/Safety: Hard limits still apply (optional, but good for safety)
        if temperature > 38.0 or gas > 800.0:
            env_anomaly = 1

        # ===========================
        # AUDIO ANOMALY CHECK (STRICT)
        # ===========================

        # MUST define audio_mse before inserting to DB
        audio_mse = abs(audio_db - 50.0) / 50.0
        
        # Model-Based Anomaly Detection
        if audio_mse > AUDIO_MSE_THRESHOLD:
            audio_anomaly = 1
        else:
            audio_anomaly = 0
            
        # Fallback: Extreme values
        if audio_db > 80 or mic_peak > 1000:
            audio_anomaly = 1

        # Humidity alert (NOT an anomaly flag)
        humidity_high = humidity > 100.0   # adjust if needed
        now = time.time()

        # Trigger alert only every 30 minutes
        # Trigger alert only every 30 minutes
        if (now - last_humidity_alert) >= HUMIDITY_ALERT_COOLDOWN:
            try:
                if humidity_high:
                    create_alert(user_id, "Humidity", f"Humidity is too high: {humidity}%", "warning")
                    last_humidity_alert = now
                    logger.info(f"🌧️ Humidity alert sent: {humidity}%")
                
                if temperature > 34.0:
                    create_alert(user_id, "Temperature", f"High Heat detected: {temperature}°C", "warning")
                    last_humidity_alert = now # Share cooldown to prevent spam
                
                if gas > 600.0:
                    create_alert(user_id, "Air Quality", f"High Ammonia/Gas levels: {gas} ppm", "danger")
                    last_humidity_alert = now

                if audio_anomaly:
                    create_alert(user_id, "Audio", f"Abnormal Audio detected (dB={audio_db})", "warning")
                    last_humidity_alert = now

            except Exception:
                logger.exception("Failed to create sensor alerts")


        # --- Save environmental data ---
        logger.info(f"Inserting env data: temp={temperature}, gas={gas}, mse={env_mse}, anom={env_anomaly}")
        insert_environmental_data(
            user_id=user_id,
            temperature=temperature,
            humidity=humidity,
            gas=gas,
            mse_score=env_mse,
            is_anomaly=env_anomaly,
            device_id=device_id
        )
        logger.info("Env data inserted successfully")

        # ✅ FIX: attach an audio file path (so frontend shows audio button)
        file_path = None
        try:
            conn, cur = _get_conn_cursor(dictionary=True)
            cur.execute(
                "SELECT file_path FROM audio_data WHERE user_id=%s "
                "AND file_path IS NOT NULL AND file_path != '' "
                "ORDER BY id DESC LIMIT 1",
                (user_id,)
            )
            row = cur.fetchone()
            file_path = row["file_path"] if row else None
            cur.close(); conn.close()
        except Exception:
            file_path = None

        # fallback filename if nothing found
        if not file_path:
            # create a stable name and create a tiny placeholder file (0 bytes) so frontend can try to fetch it
            file_name = f"auto_{int(time.time())}.wav"
            file_path = file_name
            # create the placeholder file in uploads/audio (if not exists)
            placeholder_full = AUDIO_DIR / file_name
            try:
                if not placeholder_full.exists():
                    with open(placeholder_full, "wb") as fh:
                        fh.write(b"")  # zero-byte placeholder
            except Exception:
                logger.exception("Failed to create placeholder audio file")

        # --- Save audio data ---
        insert_audio_data(
            user_id=user_id,
            rms=mic_rms,
            peak=mic_peak,
            audio_db=audio_db,
            file_path=file_path,  # ✅ added here
            mse_score=audio_mse,
            is_anomaly=audio_anomaly,
            device_id=device_id
        )

        update_device_last_seen(device_id, status="online")

        # trigger fusion immediately
        fusion_result = run_fusion_once()

        logger.info(
            f"✅ Saved from ESP32: env_mse={env_mse:.4f}, audio_mse={audio_mse:.4f}, "
            f"env_anom={env_anomaly}, audio_anom={audio_anomaly}, file={file_path}"
        )

        return jsonify({
            "message": "data processed",
            "env_mse": env_mse,
            "env_anomaly": bool(env_anomaly),
            "audio_mse": audio_mse,
            "audio_anomaly": bool(audio_anomaly),
            "file_path": file_path,
            "fusion": fusion_result
        }), 200

    except Exception as e:
        logger.exception("❌ Error saving sensor data")
        return jsonify({"error": str(e)}), 500

# -------------------------
# Upload endpoints for audio/frame
# -------------------------
@app.route("/api/upload/audio", methods=["POST"])
def api_upload_audio():
    if "file" not in request.files:
        return jsonify({"error":"file missing"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error":"file missing"}), 400
    if not allowed_file(f.filename, ALLOWED_AUDIO_EXT):
        return jsonify({"error":"file type not allowed"}), 400
    saved = save_uploaded_file(f, AUDIO_DIR)
    device_uid = request.form.get("device_uid")
    user_id = request.form.get("user_id", 1)
    # persist audio row and return filename
    try:
        insert_audio_data(user_id or 1, rms=0, peak=0, audio_db=0, file_path=os.path.basename(saved), mse_score=0.0, is_anomaly=0, device_id=device_uid)
    except Exception:
        logger.exception("Failed to insert audio upload row, continuing")
    try:
        update_device_last_seen(device_uid)
    except Exception:
        pass
    return jsonify({"status":"ok","file":os.path.basename(saved)}), 201

@app.route("/api/upload/frame", methods=["POST"])
def api_upload_frame():
    if "file" not in request.files:
        return jsonify({"error":"file missing"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error":"file missing"}), 400
    if not allowed_file(f.filename, ALLOWED_IMAGE_EXT):
        return jsonify({"error":"file type not allowed"}), 400
    saved = save_uploaded_file(f, IMG_DIR)
    device_uid = request.form.get("device_uid")
    user_id = request.form.get("user_id", 1)
    insert_vision_data(user_id or 1, behavior="Frame", bbox=None, mse_score=0.0, is_anomaly=0, frame_path=os.path.basename(saved), device_id=device_uid)
    try: update_device_last_seen(device_uid)
    except Exception: pass
    return jsonify({"status":"ok","file":os.path.basename(saved)}), 201

# -------------------------
# Serve Uploads
# -------------------------
@app.route("/uploads/<path:filename>")
def serve_uploads(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# -------------------------
# API endpoints for UI (JSON)
# -------------------------


@app.route('/api/alerts/<int:alert_id>/resolve', methods=['POST'])
@login_required
def api_resolve_alert(alert_id):
    try:
        user_id = session.get('user_id')
        rc = resolve_alert(alert_id, user_id)
        if rc > 0:
            return jsonify({"status": "ok", "message": "Alert resolved"})
        else:
            return jsonify({"error": "Alert not found or already resolved"}), 404
    except Exception as e:
        logger.exception("Failed to resolve alert")
        return jsonify({"error": str(e)}), 500

@app.route('/api/dashboard_data')
def dashboard_data():
    """Return latest environmental + audio + fusion info for dashboard."""
    try:
        conn, cur = _get_conn_cursor(dictionary=True)

        cur.execute("""
            SELECT temperature, humidity, gas, mse_score AS env_mse, timestamp
            FROM environmental_data
            WHERE user_id = 1
            ORDER BY id DESC LIMIT 1
        """)
        env = cur.fetchone()

        cur.execute("""
            SELECT rms, peak, audio_db, mse_score AS audio_mse, timestamp
            FROM audio_data
            WHERE user_id = 1
            ORDER BY id DESC LIMIT 1
        """)
        audio = cur.fetchone()

        cur.execute("""
            SELECT fusion_label, fusion_confidence
            FROM fusion_predictions
            WHERE user_id = 1
            ORDER BY id DESC LIMIT 1
        """)
        fusion = cur.fetchone()

        cur.execute("""
            SELECT mse_score AS vision_mse, is_anomaly AS vision_anomaly
            FROM vision_data
            WHERE user_id = 1
            ORDER BY id DESC LIMIT 1
        """)
        vision = cur.fetchone()

        cur.execute("""
            SELECT id, user_id, type, message, severity, timestamp, resolved
            FROM alerts
            WHERE user_id = 1 AND resolved=0
            ORDER BY timestamp DESC LIMIT 10
        """)
        alerts = cur.fetchall()

        # Fetch recent fusion history for Logs Table
        cur.execute("""
            SELECT timestamp, fusion_label, fusion_confidence, 
                   sensor_anomaly, audio_anomaly, vision_anomaly,
                   sensor_score, audio_score, vision_score
            FROM fusion_predictions
            WHERE user_id = 1
            ORDER BY timestamp DESC LIMIT 50
        """)
        recent_fusions = cur.fetchall()

        # Fetch recent vision history for Behavior Timeline
        cur.execute("""
            SELECT timestamp, behavior, mse_score, is_anomaly
            FROM vision_data
            WHERE user_id = 1
            ORDER BY timestamp DESC LIMIT 10
        """)
        recent_vision = cur.fetchall()

        # Fetch devices for Device Status Panel
        cur.execute("""
            SELECT device_uid, device_type, status, last_seen, location
            FROM devices
            WHERE user_id = 1
            ORDER BY last_seen DESC
        """)
        devices = cur.fetchall()

        cur.close()
        conn.close()

        # Helper to format timestamps
        def format_ts(rows):
            if not rows: return []
            for r in rows:
                if 'timestamp' in r and r['timestamp']:
                    if isinstance(r['timestamp'], (datetime, date)):
                        r['timestamp'] = r['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        r['timestamp'] = str(r['timestamp'])
                if 'last_seen' in r and r['last_seen']:
                    if isinstance(r['last_seen'], (datetime, date)):
                        r['last_seen'] = r['last_seen'].strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        r['last_seen'] = str(r['last_seen'])
            return rows

        alerts = format_ts(alerts)
        recent_fusions = format_ts(recent_fusions)
        recent_vision = format_ts(recent_vision)
        devices = format_ts(devices)

        # Dynamic Dashboard Suggestions
        suggestions = []
        if env:
            if env.get('temperature', 0) > 34:
                suggestions.append("🔥 High Temperature: Ensure ventilation is ON.")
            if env.get('gas', 0) > 600:
                suggestions.append("⚠️ High Ammonia: Check litter quality and airflow.")
            if env.get('humidity', 0) > 80:
                suggestions.append("💧 High Humidity: Risk of respiratory issues.")
        
        if audio and audio.get('audio_mse', 0) > 0.5:
            suggestions.append("🔊 Audio Anomaly: Check for external noise or distress.")

        if fusion and fusion.get('fusion_label', '').startswith('High'):
            suggestions.append("🚨 HIGH STRESS: Immediate inspection required!")

        if not suggestions:
            suggestions.append("✅ System Normal. No actions required.")

        return jsonify({
            'temperature': env['temperature'] if env else None,
            'humidity': env['humidity'] if env else None,
            'gas': env['gas'] if env else None,
            'mic_rms': audio['rms'] if audio else None,
            'mic_peak': audio['peak'] if audio else None,
            'audio_level': audio['audio_db'] if audio else None,
            
            'env_mse': env['env_mse'] if env else 0.0,
            'audio_mse': audio['audio_mse'] if audio else 0.0,
            'vision_mse': vision['vision_mse'] if vision else 0.0,
            
            'env_anomaly': bool(env.get('env_mse', 0) > 600) if env else False, # Using threshold logic or fetch is_anomaly if available
            'audio_anomaly': bool(audio.get('audio_mse', 0) > 0.5) if audio else False,
            'vision_anomaly': bool(vision['vision_anomaly']) if vision else False,

            'fusion_label': fusion['fusion_label'] if fusion else 'Normal',
            'fusion_confidence': fusion['fusion_confidence'] if fusion else 0.0,
            'alerts': alerts,
            'recent_fusions': recent_fusions,
            'recent_vision': recent_vision,
            'devices': devices,
            'suggestions': suggestions,
            'device_status': {'connected': True}
        })

    except Exception as e:
        import traceback
        logger.exception("❌ dashboard_data error")
        return jsonify({'error': str(e)}), 500

@app.route("/api/recent/environment/<int:user_id>")
def api_recent_environment(user_id):
    try:
        offset = int(request.args.get("offset", 0))  # pagination offset
        limit = 50

        conn, cur = _get_conn_cursor(dictionary=True)

        cur.execute("""
            SELECT id, temperature, humidity, gas, mse_score, is_anomaly, timestamp
            FROM environmental_data
            WHERE user_id=%s
            ORDER BY timestamp DESC
            LIMIT %s OFFSET %s
        """, (user_id, limit, offset))

        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return jsonify({"history": [], "page": offset})

        latest = rows[0]

        return jsonify({
            "page": offset,
            "history": [
                {
                    "timestamp": r["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                    "temperature": r["temperature"],
                    "humidity": r["humidity"],
                    "gas": r["gas"],
                    "mse_score": r["mse_score"],
                    "is_anomaly": r["is_anomaly"]
                }
                for r in rows
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/recent/audio/<int:user_id>")
def api_recent_audio(user_id):
    try:
        offset = int(request.args.get("offset", 0))
        limit = 50

        conn, cur = _get_conn_cursor(dictionary=True)

        cur.execute("""
            SELECT id, rms, peak, audio_db, mse_score, file_path, is_anomaly, timestamp
            FROM audio_data
            WHERE user_id=%s
            ORDER BY timestamp DESC
            LIMIT %s OFFSET %s
        """, (user_id, limit, offset))

        rows = cur.fetchall()
        cur.close()
        conn.close()

        return jsonify([
            {
                "timestamp": r["timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
                "rms": r["rms"],
                "peak": r["peak"],
                "audio_db": r["audio_db"],
                "mse_score": r["mse_score"],
                "file_path": r["file_path"],
                "is_anomaly": r["is_anomaly"]
            } for r in rows
        ])

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/audio_data")
def api_audio_data():
    """Return current and historical audio analysis data for the dashboard."""
    try:
        conn, cur = _get_conn_cursor(dictionary=True)

        # Latest record
        cur.execute("""
            SELECT rms, peak, audio_db, mse_score, timestamp, is_anomaly, file_path
            FROM audio_data
            WHERE user_id = 1
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        latest = cur.fetchone()

        # Recent 20 records
        cur.execute("""
            SELECT rms, peak, audio_db, mse_score, is_anomaly, timestamp, file_path
            FROM audio_data
            WHERE user_id = 1
            ORDER BY timestamp DESC
            LIMIT 20
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Placeholder waveform & spectrogram (you can later generate from raw audio)
        waveform = [{"time": i, "value": float(np.sin(i/3)*1000)} for i in range(30)]
        spectrogram = [{"freq": f, "intensity": int(np.random.randint(10, 100))} for f in range(15)]

        # Dynamic Suggestions
        suggestions = []
        if latest:
            if latest["is_anomaly"]:
                suggestions.append("⚠️ Audio Anomaly Detected: Check for predators or equipment noise.")
                if latest["audio_db"] > 80:
                    suggestions.append("🔊 High Noise Level (>80dB): Inspect ventilation fans or machinery.")
                if latest["peak"] > 1000:
                    suggestions.append("💥 Sudden loud noise detected: Possible stress event.")
            else:
                suggestions.append("✅ Audio levels are within normal range.")
                
            if latest["audio_db"] < 30:
                suggestions.append("ℹ️ Environment is very quiet.")
        else:
            suggestions.append("Waiting for audio data...")

        # Format for frontend
        return jsonify({
            "current": {
                "rms": latest["rms"] if latest else None,
                "peak": latest["peak"] if latest else None,
                "db": latest["audio_db"] if latest else None,
                "mse": latest["mse_score"] if latest else None,
                "time": latest["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if latest and isinstance(latest["timestamp"], datetime) else (latest["timestamp"] if latest else None)
            },
            "anomalies": [
                {
                    "time": r["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if isinstance(r["timestamp"], datetime) else r["timestamp"],
                    "rms": r["rms"],
                    "peak": r["peak"],
                    "db": r["audio_db"],
                    "mse": r["mse_score"],
                    "label": "High" if r["is_anomaly"] else "Normal",
                    "suggestion": "Check for loud noises" if r["is_anomaly"] else "Normal",
                    "file": r["file_path"] or ""
                } for r in rows
            ],
            "waveform": waveform,
            "spectrogram": spectrogram,
            "suggestions": suggestions
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# =========================================
# 👁️ Vision Data Endpoints & Live Stream
# =========================================

@app.route("/api/recent/vision/<int:user_id>")
def api_recent_vision(user_id):
    conn, cur = _get_conn_cursor(dictionary=True)
    cur.execute("""
        SELECT id, user_id, timestamp, behavior, bbox, mse_score, is_anomaly, frame_path
        FROM vision_data
        WHERE user_id = %s
        ORDER BY timestamp DESC
        LIMIT 20
    """, (user_id,))
    data = cur.fetchall()
    cur.close(); conn.close()
    return jsonify(data)



# -------------------------
# Vision live stream (OpenCV + YOLO + LSTM) — Fixed and Database-Integrated
# -------------------------
behavior_label = "Normal"
behavior_confidence = 1.0
behavior_visible_until = 0.0
feature_sequence = []
SEQUENCE_LENGTH = int(os.environ.get("SEQUENCE_LENGTH", 8))
latest_anomaly = None
status_alert = False
last_anomaly_bbox = None
last_anomaly_time = 0.0
ANOMALY_COOLDOWN = int(os.environ.get("ANOMALY_COOLDOWN", 10))
last_resting_time = 0
RESTING_COOLDOWN = 60  # 1 minute
latest_sensor_flag = 0
latest_audio_flag = 0

# -------------------------
# Camera setup
# -------------------------
def find_working_camera(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            logger.info("[VISION] Camera %s opened", i)
            return cap, i
        else:
            try:
                cap.release()
            except Exception:
                pass
    logger.warning("[VISION] No webcam found")
    return None, None

camera, cam_index = find_working_camera()
cameras = {cam_index: camera} if camera else {}
def release_cameras():
    for cap in cameras.values():
        try:
            cap.release()
        except Exception:
            pass
atexit.register(release_cameras)

latest_frame = None
annotated_frame = None
camera_active = False
vision_active = False 
camera_lock = threading.Lock()

def background_camera_loop():
    """Continuously grab frames from webcam and store latest one."""
    global latest_frame, camera, cam_index
    logger.info(f"[VISION] Background camera loop started on index {cam_index}")
    fail_count = 0
    while True:
        if camera is None or not camera.isOpened():
            time.sleep(2)
            # Try to reconnect
            try:
                camera, cam_index = find_working_camera()
            except:
                pass
            continue
        
        try:
            ret, frame = camera.read()
            if ret:
                fail_count = 0
                with camera_lock:
                    latest_frame = frame.copy()
            else:
                fail_count += 1
                if fail_count > 10:
                    logger.warning("[VISION] Camera failing to read, releasing...")
                    camera.release()
                    camera = None
                    fail_count = 0
                time.sleep(0.1)
        except Exception:
            logger.exception("[VISION] Camera read error")
            time.sleep(1)
        
        time.sleep(0.03)  # ~30 fps

# -------------------------
# Ensure models are loaded
# -------------------------
def _ensure_models_loaded_for_vision():
    _load_yolo()
    _load_tf_models()

# -------------------------
# Database insert helper
# -------------------------
def save_to_vision_data(behavior, mse_score, bbox, is_anomaly):
    """Insert vision result into vision_data, then trigger fusion using this exact anomaly flag."""
    try:
        conn, cur = _get_conn_cursor()
        query = """
        INSERT INTO vision_data (user_id, timestamp, behavior, bbox, mse_score, is_anomaly, device_id, frame_path)
        VALUES (%s, NOW(), %s, %s, %s, %s, %s, %s)
        """
        cur.execute(query, (1, behavior, json.dumps(bbox), mse_score, is_anomaly, "esp32_001", "-"))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"[VISION_DB] Saved: {behavior}, mse={mse_score:.2f}, anomaly={is_anomaly}")
        if is_anomaly:
             logger.info(f"[VISION_TRIGGER] Anomaly Triggered by: {behavior} (MSE={mse_score:.2f})")

        # 🔄 Immediately trigger fusion using THIS anomaly info
        try:
            # Manually pass the latest vision anomaly + mse into fusion call
            sensor_row, audio_row, _ = fetch_latest_rows_all()
            sensor_score  = float(sensor_row.get("sensor_score") or 0.0) if sensor_row else 0.0
            sensor_flag   = int(bool(sensor_row.get("sensor_flag"))) if sensor_row else 0
            audio_score   = float(audio_row.get("audio_score") or 0.0) if audio_row else 0.0
            audio_flag    = int(bool(audio_row.get("audio_flag"))) if audio_row else 0

            # use current vision values directly
            vision_score  = float(mse_score)
            vision_flag   = int(bool(is_anomaly))

            label, conf = call_fusion(sensor_score, audio_score, vision_score,
                                      sensor_flag, audio_flag, vision_flag)

            fid = insert_fusion_prediction(
                1, sensor_flag, audio_flag, vision_flag,
                sensor_score, audio_score, vision_score, label, conf
            )
            logger.info(f"[FUSION_TRIGGER] Vision anomaly({vision_flag}) inserted fusion id={fid}, label={label}, conf={conf:.2f}")
        except Exception:
            logger.exception("[FUSION_TRIGGER] Fusion run failed after vision save")

    except Exception as e:
        logger.exception(f"[VISION_DB ERROR] {e}")
# Helper: Save tracked chicken
# -------------------------
# Chicken Tracker Class
# -------------------------
class ChickenTracker:
    def __init__(self):
        self.tracks = {}  # {id: {state, first_stress, last_seen, bbox, recovered}}
        self.next_id = 1
        self.STRESS_THRESHOLD = 0  # seconds (Immediate tracking)
        self.MATCH_DIST = 150  # pixels
        self.CLEANUP_TIMEOUT = 300  # 5 minutes

    def update(self, detections, frame, sensor_flag, audio_flag, vision_flag=0):
        now = time.time()
        
        # 1. Match detections to existing tracks
        # detections: list of {class, conf, bbox}
        # simple greedy matching
        
        active_detections = [] # (det, track_id)
        used_tracks = set()
        
        for det in detections:
            best_id = None
            min_dist = float('inf')
            cx = (det['bbox'][0] + det['bbox'][2]) / 2
            cy = (det['bbox'][1] + det['bbox'][3]) / 2
            
            for tid, track in self.tracks.items():
                if tid in used_tracks: continue
                tcx = (track['bbox'][0] + track['bbox'][2]) / 2
                tcy = (track['bbox'][1] + track['bbox'][3]) / 2
                dist = np.linalg.norm([cx - tcx, cy - tcy])
                
                if dist < self.MATCH_DIST and dist < min_dist:
                    min_dist = dist
                    best_id = tid
            
            if best_id:
                used_tracks.add(best_id)
                self._update_track(best_id, det, now, frame, sensor_flag, audio_flag, vision_flag)
            else:
                self._create_track(det, now, frame, sensor_flag, audio_flag, vision_flag)

        # 2. Handle missing tracks (check for recovery or cleanup)
        for tid in list(self.tracks.keys()):
            if tid not in used_tracks:
                track = self.tracks[tid]
                # If not seen for a while, cleanup
                if now - track['last_seen'] > self.CLEANUP_TIMEOUT:
                    del self.tracks[tid]
                    continue
                
                # If it was active/monitoring and now missing (neutral/gone), 
                # we don't immediately mark recovered unless we get explicit neutral detection.
                # But here we assume missing = neutral/gone if it was stress-based tracking.
                pass

    def _create_track(self, det, now, frame, sensor_flag, audio_flag, vision_flag):
        # Only create track if stress/anomaly detected (Vision, Sensor, or Audio)
        # MODIFIED: Allow vision-only stress to trigger tracking
        if det['class'] in ('stress', 'anomaly'):
            # if not (sensor_flag or audio_flag): return  <-- REMOVED restriction


            tid = f"Chicken {self.next_id}"
            self.next_id += 1
            self.tracks[tid] = {
                'state': 'monitoring',
                'first_stress': now,
                'last_seen': now,
                'last_anomaly_time': now, # Track when last anomaly occurred
                'bbox': det['bbox'],
                'recovered': False
            }
            logger.info(f"[TRACKER] New potential stress track: {tid} (VisionStress=1)")

        # ALSO trigger if global vision flag (behavior) is set, even if YOLO is neutral?
        # For now, we only start *individual* tracks if YOLO says "stress" OR if we decide to upgrade neutrals.
        # Let's upgrade Neutral to Stress if vision_flag is True (Behavior Anomaly)
        if vision_flag and det['class'] in ('neutral', 'normal', 'healthy'):
             tid = f"Chicken {self.next_id}"
             self.next_id += 1
             self.tracks[tid] = {
                'state': 'monitoring',
                'first_stress': now,
                'last_seen': now,
                'last_anomaly_time': now,
                'bbox': det['bbox'],
                'recovered': False
             }
             logger.info(f"[TRACKER] New behavior-based track: {tid} (Behavior Anomaly)")
        
        # NEW: Allow tracking of NEUTRAL chickens (Green Boxes)
        elif det['class'] in ('neutral', 'normal', 'healthy'):
             tid = f"Chicken {self.next_id}"
             self.next_id += 1
             self.tracks[tid] = {
                'state': 'monitoring',
                'first_stress': 0, # Not stressed
                'last_seen': now,
                'last_anomaly_time': 0,
                'bbox': det['bbox'],
                'recovered': True # Mark as recovered/normal initially
             }

    def _update_track(self, tid, det, now, frame, sensor_flag, audio_flag, vision_flag):
        track = self.tracks[tid]
        track['last_seen'] = now
        track['bbox'] = det['bbox']
        
        is_vision_stress = det['class'] in ('stress', 'anomaly')
        any_anomaly = is_vision_stress or sensor_flag or audio_flag or vision_flag
        
        if any_anomaly:
            track['last_anomaly_time'] = now
            
            # If currently monitoring or active, we confirm/maintain active
            if track['state'] == 'monitoring':
                # Check threshold (now 0s, so immediate)
                if now - track['first_stress'] >= self.STRESS_THRESHOLD:
                    track['state'] = 'active'
                    track['recovered'] = False
                    self._save_to_db(tid, track, frame, det['conf'], sensor_flag, audio_flag)
                    logger.info(f"[TRACKER] {tid} confirmed STRESS (Active) [Vision=1, Sensor={sensor_flag}, Audio={audio_flag}]")
            
            elif track['state'] == 'active':
                track['recovered'] = False
                # Update DB every 10s
                if now - track.get('last_db_update', 0) > 10:
                    self._save_to_db(tid, track, frame, det['conf'], sensor_flag, audio_flag)
            
            elif track['state'] == 'recovered':
                # Relapse
                track['state'] = 'active'
                track['recovered'] = False
                self._save_to_db(tid, track, frame, det['conf'], sensor_flag, audio_flag)
                logger.info(f"[TRACKER] {tid} RELAPSE -> Active [Vision=1, Sensor={sensor_flag}, Audio={audio_flag}]")

        else: 
            # All Normal (Vision Neutral + No Sensor/Audio Anomaly)
            # Check for recovery buffer (5 seconds)
            if track['state'] == 'active':
                if now - track['last_anomaly_time'] > 5.0:
                    track['state'] = 'recovered'
                    track['recovered'] = True
                    self._save_to_db(tid, track, frame, det['conf'], sensor_flag, audio_flag)
                    logger.info(f"[TRACKER] {tid} Recovered (Strict)")
            
            elif track['state'] == 'monitoring':
                # False alarm, reset or delete
                del self.tracks[tid]

    def _save_to_db(self, tid, track, frame, conf, sensor_flag, audio_flag):
        mse = round((1 - conf) * 1000, 4)
        bbox_str = f"{track['bbox'][0]},{track['bbox'][1]},{track['bbox'][2]},{track['bbox'][3]}"
        
        # Determine label
        label = "Stress"
        if sensor_flag and audio_flag:
            label = "High Stress"
        elif sensor_flag:
            label = "Stress (Env)"
        elif audio_flag:
            label = "Stress (Audio)"
            
        if track['recovered']:
            label = "Recovered"

        # Use existing helper but pass our simple ID
        _save_tracked_chicken(
            tracked_id=tid,
            frame=frame,
            bbox=bbox_str,
            mse_score=mse,
            notes=f"{label}"
        )
        track['last_db_update'] = time.time()
        
        # Also update recovered status in DB if needed
        if track['recovered']:
             try:
                conn, cur = _get_conn_cursor()
                cur.execute("UPDATE tracked_chickens SET recovered=1, recovery_timestamp=NOW() WHERE tracked_id=%s AND user_id=1", (tid,))
                conn.commit(); cur.close(); conn.close()
             except: pass
        else:
             try:
                conn, cur = _get_conn_cursor()
                cur.execute("UPDATE tracked_chickens SET recovered=0 WHERE tracked_id=%s AND user_id=1", (tid,))
                conn.commit(); cur.close(); conn.close()
             except: pass

tracker = ChickenTracker()

# -------------------------
# Continuous YOLO + LSTM Inference
# -------------------------
def continuous_vision_inference():
    global annotated_frame, feature_sequence, behavior_label, behavior_confidence, behavior_visible_until
    global latest_anomaly, status_alert, latest_sensor_flag, latest_audio_flag

    _ensure_models_loaded_for_vision()
    logger.info("[VISION] YOLO+LSTM background inference started")

    while True:
        if latest_frame is None:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for camera...", (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            annotated_frame = blank
            time.sleep(0.2)
            continue

        with camera_lock:
            frame = latest_frame.copy()

        detections = []
        names_src = {}
        res = None

        # -------------------------
        # YOLO Inference
        # -------------------------
        try:
            if yolo_model:
                res = yolo_model(frame, verbose=False)
                detections = getattr(res[0], "boxes", []) or []
                names_src = getattr(res[0], "names", {}) or {}
        except Exception:
            logger.exception("YOLO inference error")
            detections = []

        # Prepare detections for tracker
        tracker_dets = []
        yolo_stress_detected = False
        
        for box in detections:
            try:
                cls_name = names_src.get(int(box.cls[0]), str(int(box.cls[0]))).lower()
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2, y2)
                
                if cls_name in ("stress", "anomaly") and conf >= 0.85:
                    tracker_dets.append({'class': 'stress', 'conf': conf, 'bbox': bbox})
                    yolo_stress_detected = True
                elif cls_name in ("neutral", "normal", "healthy") and conf >= 0.60:
                    tracker_dets.append({'class': 'neutral', 'conf': conf, 'bbox': bbox})
            except:
                continue
        
        # Update Tracker with latest flags (including global behavior anomaly from PREVIOUS frame or current logic)
        # We use 'latest_anomaly' which is updated by the LSTM block below (so it's 1 frame lag, which is fine)
        current_vision_flag = 1 if (latest_anomaly or behavior_label in ["Pecking", "Pacing"]) else 0
        tracker.update(tracker_dets, frame, latest_sensor_flag, latest_audio_flag, vision_flag=current_vision_flag)

        # -------------------------
        # LSTM Behavior Prediction (Legacy/Parallel)
        # -------------------------
        # (Kept for behavior logging, but not interfering with stress tracking)
        try:
            if detections and cnn_model is not None and lstm_model is not None:
                largest = max(detections, key=lambda b: (b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]))
                x1, y1, x2, y2 = map(int, largest.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_r = cv2.resize(crop, (224, 224))
                    x = keras_image.img_to_array(crop_r)
                    x = np.expand_dims(x, 0)
                    x = preprocess_input(x)
                    feat = cnn_model.predict(x, verbose=0)
                    feature_sequence.append(feat.flatten())

            if lstm_model is not None and len(feature_sequence) >= SEQUENCE_LENGTH:
                seq = np.expand_dims(np.array(feature_sequence[-SEQUENCE_LENGTH:]), axis=0)
                pred = lstm_model.predict(seq, verbose=0)[0]
                if len(pred) == 2:
                    p_pacing, p_pecking = float(pred[0]), float(pred[1])
                else:
                    p_pecking = float(pred[-1])
                    p_pacing = 1.0 - p_pecking

                new_label, new_conf = None, 0.0
                if p_pecking >= 0.6:
                    new_label, new_conf = "Pecking", p_pecking
                elif p_pacing >= 0.6:
                    new_label, new_conf = "Pacing", p_pacing

                if new_label:
                    behavior_label = new_label
                    behavior_confidence = new_conf
                    behavior_visible_until = time.time() + 10
                    feature_sequence = []
                    
                    # Calculate base MSE from behavior confidence
                    # Normal: conf ~0.9 -> mse ~100
                    # Anomaly: conf ~0.8 -> mse ~200
                    base_mse = round((1-new_conf)*1000, 4)
                    
                    # Model-Based Anomaly Detection
                    # If YOLO says stress, we force MSE to be high enough to trigger anomaly
                    if yolo_stress_detected:
                        final_mse = max(base_mse, VISION_MSE_THRESHOLD + 50.0) # e.g. 250.0
                    else:
                        final_mse = base_mse

                    # Determine anomaly based on MSE threshold AND "Many Chickens" Rule
                    # Rule: Anomaly = 1 if ANY chicken is stressed (YOLO) OR Behavior is persistent
                    stressed_count = len([d for d in tracker_dets if d['class'] == 'stress'])
                    
                    is_anomaly = 0
                    if (final_mse > VISION_MSE_THRESHOLD or latest_sensor_flag or latest_audio_flag):
                        # Enforce "Many Chickens" rule for pure vision triggers
                        # RELAXED: Allow even 1 stressed chicken to trigger
                        if stressed_count >= 1 or behavior_label in ["Pecking", "Pacing"]:
                             is_anomaly = 1
                        # If sensor/audio is already anomalous, we can be more lenient
                        elif latest_sensor_flag or latest_audio_flag:
                             is_anomaly = 1
                    
                    # Log behavior but don't create new tracks here to avoid conflict
                    save_to_vision_data(behavior_label, final_mse, "-", is_anomaly)
                    
        except Exception:
            pass

        # -------------------------
        # Draw annotations
        # -------------------------
        annotated = frame.copy()

        # FORCE DRAW ALL DETECTIONS (Dynamic Color Coding)
        for box in detections:
            try:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_name = names_src.get(int(box.cls[0]), str(int(box.cls[0]))).lower()
                
                # Default Green (Normal)
                color = (0, 255, 0) 
                display_label = f"{cls_name} {float(box.conf[0]):.2f}"
                
                if cls_name in ("stress", "anomaly"):
                    # Check for ANY other modality (Sensor OR Audio)
                    if latest_sensor_flag or latest_audio_flag:
                        # Confirmed Stress: Red
                        color = (0, 0, 255)
                        display_label = f"High Stress {float(box.conf[0]):.2f}"
                    else:
                        # Vision Only: Yellow
                        color = (0, 255, 255) 
                        display_label = f"Behavior Anomaly {float(box.conf[0]):.2f}"
                
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, display_label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except: pass

        # Draw Tracker Info (Vanishable: Only if seen recently)
        for tid, track in tracker.tracks.items():
            # Only draw if seen in the last 1.0 second
            if (time.time() - track['last_seen']) > 1.0:
                continue

            if track['state'] == 'active':
                x1, y1, x2, y2 = track['bbox']
                
                if track['recovered']:
                    color = (0, 255, 0)
                    label = f"{tid} (Rec)"
                else:
                    # Check fusion flags for severity
                    if latest_sensor_flag or latest_audio_flag:
                        color = (0, 0, 255) # Red
                        label = f"{tid} (High Stress)"
                    else:
                        color = (0, 255, 255) # Yellow
                        label = f"{tid} (Behavior Anomaly)"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            elif track['state'] == 'monitoring':
                x1, y1, x2, y2 = track['bbox']
                elapsed = int(time.time() - track['first_stress'])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 1)
                cv2.putText(annotated, f"Mon: {elapsed}s", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        annotated_frame = annotated.copy()
        time.sleep(0.03)

# ==========================================
def _save_tracked_chicken(tracked_id, frame, bbox, mse_score, notes=""):
    try:
        # Parse bbox
        x1, y1, x2, y2 = 0, 0, frame.shape[1], frame.shape[0]
        if isinstance(bbox, str) and "," in bbox:
            try:
                x1, y1, x2, y2 = map(int, bbox.split(","))
            except:
                pass
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)

        # Ensure bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Crop if valid
        if x2 > x1 and y2 > y1:
            crop = frame[y1:y2, x1:x2]
        else:
            crop = frame

        conn, cur = _get_conn_cursor()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save cropped image to IMG_DIR (uploads/images) so frontend can serve it via /uploads/images/...
        # Filename: tracked_{id}_{timestamp}.jpg
        filename = f"tracked_{tracked_id}_{timestamp}.jpg"
        img_path = IMG_DIR / filename
        
        # os.makedirs(os.path.dirname(img_path), exist_ok=True) # IMG_DIR already exists
        cv2.imwrite(str(img_path), crop)

        # Store ONLY filename in DB if it's not already
        bbox_str = bbox if isinstance(bbox, str) else json.dumps(bbox)

        cur.execute("""
            INSERT INTO tracked_chickens 
                (user_id, tracked_id, frame_path, vision_bbox, mse_score, first_seen, last_seen, recovered, notes)
            VALUES 
                (%s, %s, %s, %s, %s, NOW(), NOW(), 0, %s)
            ON DUPLICATE KEY UPDATE
                last_seen = NOW(),
                mse_score = VALUES(mse_score),
                frame_path = VALUES(frame_path),
                vision_bbox = VALUES(vision_bbox),
                notes = VALUES(notes)
        """, (1, tracked_id, filename, bbox_str, mse_score, notes))

        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"[DB] Tracked chicken saved: {tracked_id} (Cropped)")
    except Exception:
        logger.exception("[DB] Failed to save tracked chicken")





# MJPEG stream endpoint uses the annotated_frame created by continuous thread
@app.route("/vision_stream")
@login_required
def vision_stream():
    def frame_generator():
        global annotated_frame
        while True:
            if annotated_frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Initializing Vision Feed.", (60, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                _, buf = cv2.imencode(".jpg", blank)
            else:
                _, buf = cv2.imencode(".jpg", annotated_frame)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.03)
    return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")
# ==========================================
# 🧠 FUSION LOGIC + CHICKEN TRACKING MODULE
# ==========================================
import os, cv2, time, json
import numpy as np
from flask import jsonify

# 🔹 FUSION: combine latest sensor, audio, and vision signals
def run_fusion_once():
    """Runs fusion and triggers chicken tracking when stress detected."""
    try:
        conn, cur = _get_conn_cursor()

        # --- Fetch latest data from each source
        cur.execute("SELECT id, user_id, mse_score AS sensor_score, is_anomaly AS sensor_flag FROM environmental_data ORDER BY id DESC LIMIT 1")
        sensor_row = cur.fetchone()

        cur.execute("SELECT id, user_id, mse_score AS audio_score, is_anomaly AS audio_flag FROM audio_data ORDER BY id DESC LIMIT 1")
        audio_row = cur.fetchone()

        cur.execute("""
            SELECT id, user_id, mse_score AS vision_score, is_anomaly AS vision_flag,
                   bbox AS vision_bbox, frame_path, timestamp
            FROM vision_data
            ORDER BY id DESC LIMIT 1
        """)
        vision_row = cur.fetchone()

        # --- Prepare values
        sensor_score = float(sensor_row["sensor_score"]) if sensor_row else 0
        sensor_flag  = int(sensor_row["sensor_flag"]) if sensor_row else 0
        audio_score  = float(audio_row["audio_score"]) if audio_row else 0
        audio_flag   = int(audio_row["audio_flag"]) if audio_row else 0
        vision_score = float(vision_row["vision_score"]) if vision_row else 0
        vision_flag  = int(vision_row["vision_flag"]) if vision_row else 0

        # Update globals for vision tracker
        global latest_sensor_flag, latest_audio_flag
        latest_sensor_flag = sensor_flag
        latest_audio_flag = audio_flag

        # --- Run fusion model
        label, conf = call_fusion(sensor_score, audio_score, vision_score,
                                  sensor_flag, audio_flag, vision_flag)

        fid = insert_fusion_prediction(
            1, sensor_flag, audio_flag, vision_flag,
            sensor_score, audio_score, vision_score, label, conf
        )
        logger.info(f"[FUSION] label={label}, conf={conf:.2f}")

        # 🟩 TRACKING: Handled by continuous_vision_inference
        # if label.lower() == "high stress" and vision_row:
        #     # Tracking logic removed to avoid duplication and invalid file path errors.
        #     pass

        cur.close()
        conn.close()
        return {"fusion_id": fid, "label": label, "confidence": conf}

    except Exception as e:
        logger.exception(f"[FUSION ERROR] {e}")
        return None


# ==========================================
# 🐔 HELPER: Update 'last_seen' when re-detected
# ==========================================
def update_chicken_last_seen(tracked_id, mse_score):
    """Update last seen timestamp and MSE."""
    try:
        conn, cur = _get_conn_cursor()
        cur.execute("""
            UPDATE tracked_chickens
            SET last_seen = NOW(), mse_score = %s
            WHERE tracked_id = %s
        """, (mse_score, tracked_id))
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"[TRACKING UPDATE] Last seen updated for {tracked_id}")
    except Exception as e:
        logger.exception(f"[TRACKING UPDATE ERROR] {e}")


# ==========================================
# 🧩 API ENDPOINT: /api/tracked_chickens
# ==========================================
@app.route("/api/tracked_chickens")
def api_tracked_chickens():
    """
    Returns recently tracked chickens with latest MSE, photo, and recovery state.
    Auto-marks chickens as recovered if no new stress for 3 minutes.
    """
    try:
        conn, cur = _get_conn_cursor()

        # 🔹 Auto-mark recovered
        cur.execute("""
            UPDATE tracked_chickens
            SET recovered = 1, recovery_timestamp = NOW(), notes = 'Recovered after inactivity'
            WHERE recovered = 0 AND TIMESTAMPDIFF(MINUTE, last_seen, NOW()) >= 3
        """)
        conn.commit()

        # 🔹 Fetch recent tracked entries
        cur.execute("""
            SELECT 
                tracked_id,
                frame_path,
                mse_score,
                DATE_FORMAT(last_seen, '%Y-%m-%d %H:%i:%s') AS last_seen,
                recovered
            FROM tracked_chickens
            ORDER BY last_seen DESC
            LIMIT 6
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        data = []
        for r in rows:
            data.append({
                "tracked_id": r["tracked_id"],
                "frame_path": r["frame_path"],
                "mse_score": float(r["mse_score"]) if r["mse_score"] else 0.0,
                "last_seen": r["last_seen"],
                "recovered": int(r["recovered"]),
            })
        return jsonify(data)

    except Exception as e:
        logger.exception("[API] tracked_chickens error")
        return jsonify([])

# -------------------------
# Health
# -------------------------
@app.route("/health")
def health_check():
    ok = {"app": True, "models": {"yolo": bool(yolo_model), "cnn": bool(cnn_model), "lstm": bool(lstm_model), "audio": bool(audio_model)}, "database": False}
    try:
        conn = db_connect(); cur = conn.cursor(); cur.execute("SELECT 1"); cur.close(); conn.close()
        ok["database"] = True
    except Exception as e:
        ok["db_error"] = str(e)
    return jsonify(ok)

# -------------------------
# Analytics Endpoint (FIXED)
# -------------------------
@app.route("/api/analytics_data")
@login_required
def api_analytics_data():
    user_id = session.get("user_id", 1)

    try:
        conn, cur = _get_conn_cursor(dictionary=True)

        # ----------------------------------------------------
        # 1) SUMMARY COUNTS
        # ----------------------------------------------------
        cur.execute("""
            SELECT fusion_label, COUNT(*) AS cnt, AVG(fusion_confidence) AS avg_conf
            FROM fusion_predictions
            WHERE user_id=%s
            GROUP BY fusion_label
        """, (user_id,))
        rows = cur.fetchall()

        summary = {"normal_count": 0, "mild_count": 0, "high_count": 0, "avg_conf": 0.0}
        total_conf = 0; total_cnt = 0

        for r in rows:
            lbl = (r["fusion_label"] or "").lower()
            cnt = r["cnt"]
            avg_conf = r["avg_conf"]

            if "high" in lbl:
                summary["high_count"] += cnt
            elif "mild" in lbl:
                summary["mild_count"] += cnt
            else:
                summary["normal_count"] += cnt

            total_conf += avg_conf * cnt
            total_cnt += cnt

        summary["avg_conf"] = round(total_conf / total_cnt, 3) if total_cnt else 0.0

        # ----------------------------------------------------
        # 2) WEEKLY EGG PRODUCTION
        # ----------------------------------------------------
        cur.execute("""
            SELECT YEARWEEK(date, 1) AS week_no,
                   SUM(total_eggs) AS total_eggs
            FROM egg_production
            WHERE user_id=%s
            GROUP BY YEARWEEK(date, 1)
            ORDER BY week_no ASC
        """, (user_id,))
        trends = cur.fetchall()

        # ----------------------------------------------------
        # 3) HEATMAP (USE REAL COLUMN NAMES)
        # ----------------------------------------------------
        cur.execute("""
            SELECT 
                AVG(sensor_score) AS sensor_score,
                AVG(audio_score) AS audio_score,
                AVG(vision_score) AS vision_score
            FROM fusion_predictions
            WHERE user_id=%s
        """, (user_id,))
        corr = cur.fetchone() or {}

        sensor = float(corr.get("sensor_score") or 0.0)
        audio  = float(corr.get("audio_score") or 0.0)
        vision = float(corr.get("vision_score") or 0.0)

        heatmap = [
            {"x": "Sensor", "y": "Sensor", "v": sensor},
            {"x": "Sensor", "y": "Audio",  "v": (sensor + audio) / 2},
            {"x": "Sensor", "y": "Vision", "v": (sensor + vision) / 2},

            {"x": "Audio", "y": "Sensor", "v": (audio + sensor) / 2},
            {"x": "Audio", "y": "Audio",  "v": audio},
            {"x": "Audio", "y": "Vision", "v": (audio + vision) / 2},

            {"x": "Vision", "y": "Sensor", "v": (vision + sensor) / 2},
            {"x": "Vision", "y": "Audio",  "v": (vision + audio) / 2},
            {"x": "Vision", "y": "Vision", "v": vision}
        ]

        # ----------------------------------------------------
        # 4) RECENT RECORDS TABLE (USE REAL COLUMNS)
        # ----------------------------------------------------
        cur.execute("""
            SELECT timestamp,
                   sensor_score,
                   audio_score,
                   vision_score,
                   fusion_label,
                   fusion_confidence
            FROM fusion_predictions
            WHERE user_id=%s
            ORDER BY timestamp DESC
            LIMIT 50
        """, (user_id,))
        recent = cur.fetchall()

        cur.close()
        conn.close()

        return jsonify({
            "summary": summary,
            "trends": trends,
            "heatmap": heatmap,
            "recent": recent
        })

    except Exception:
        logger.exception("api_analytics_data failed")
        return jsonify({"error": "failed"}), 500



# -------------------------
# Egg endpoints
# -------------------------
@app.route("/api/egg_data")
@login_required
def api_egg_data():
    user_id = session.get("user_id", 1)
    try:
        # --- DAILY (already correct) ---
        daily = fetch_egg_production(user_id, limit=30)

        # --- WEEKLY (fix: use YEARWEEK for accuracy) ---
        conn, cur = _get_conn_cursor(dictionary=True)
        cur.execute("""
            SELECT YEARWEEK(date, 1) AS week_no,
                   SUM(total_eggs) AS total_eggs
            FROM egg_production
            WHERE user_id=%s
            GROUP BY YEARWEEK(date, 1)
            ORDER BY week_no ASC
        """, (user_id,))
        weekly = cur.fetchall()

        # --- CORRELATION ---
        cur.execute("""
            SELECT AVG(
                     CASE avg_stress_label
                        WHEN 'High' THEN 2
                        WHEN 'Mild' THEN 1
                        ELSE 0
                     END
                   ) AS stress_index,
                   AVG(total_eggs) AS avg_eggs
            FROM egg_production
            WHERE user_id=%s
        """, (user_id,))
        row = cur.fetchone()
        
        corr = {
            "stress_index": float(row["stress_index"] or 0.0) if row else 0.0,
            "avg_eggs": float(row["avg_eggs"] or 0.0) if row else 0.0
        }

        cur.close()
        conn.close()

        return jsonify({
            "daily": daily,
            "weekly": weekly,
            "correlation": corr
        })

    except Exception:
        logger.exception("api_egg_data failed")
        return jsonify({"error": "failed"}), 500



@app.route("/api/add_egg_entry", methods=["POST"])
@login_required
def api_add_egg_entry():
    try:
        user_id = session.get("user_id", 1)
        payload = request.get_json(force=True)
        date_val = payload.get("date")
        eggs = int(payload.get("eggs", 0))
        broken = int(payload.get("broken", 0))

        # AVG STRESS for that date (FIXED LOGIC)
        # Instead of averaging confidence (which is high for Normal too),
        # we check if there were ANY stress events.
        conn, cur = _get_conn_cursor(dictionary=True)
        cur.execute("""
            SELECT fusion_label, COUNT(*) as cnt
            FROM fusion_predictions
            WHERE DATE(timestamp)=%s AND user_id=%s
            GROUP BY fusion_label
        """, (date_val, user_id))
        rows = cur.fetchall()
        
        has_high = False
        has_mild = False
        
        for r in rows:
            lbl = (r["fusion_label"] or "").lower()
            if "high" in lbl: has_high = True
            if "mild" in lbl or "alert" in lbl or "anomaly" in lbl: has_mild = True
            
        if has_high:
            label = "High"
        elif has_mild:
            label = "Mild"
        else:
            label = "Normal"

        insert_egg_production(user_id, date_val, eggs, broken, label, notes=None)

        return jsonify({"status": "ok", "avg_stress_label": label})

    except Exception:
        logger.exception("add egg entry failed")
        return jsonify({"error": "failed"}), 500


# -------------------------
# Convenience: start background threads once
# -------------------------
# Camera background
if camera and not camera_active:
    t_cam = threading.Thread(target=background_camera_loop, daemon=True)
    t_cam.start()
    logger.info("Background camera thread started")

# Vision inference background
if not vision_active:
    # make sure models loaded lazily
    _load_yolo(); _load_tf_models(); _load_fusion_model()
    vision_thread = threading.Thread(target=continuous_vision_inference, daemon=True)
    vision_thread.start()
    vision_active = True
    logger.info("Vision inference background thread started")

# Fusion background thread
fusion_thread = threading.Thread(target=fusion_background_loop, args=(FUSION_INTERVAL,), daemon=True)
fusion_thread.start()
logger.info("Fusion background thread started")

# -------------------------
# App runner
# -------------------------
def main():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = bool(int(os.environ.get("DEBUG", "0")))
    logger.info("Starting Flask app on %s:%s debug=%s", host, port, debug)
    logger.info("MODELS LOADED: YOLO=%s, CNN=%s, LSTM=%s, AUDIO=%s, FUSION=%s", 
                bool(yolo_model), bool(cnn_model), bool(lstm_model), bool(audio_model), bool(fusion_model))
    app.run(host=host, port=port, debug=debug, threaded=True)

# -------------------------
# History 
# -------------------------
@app.route('/api/history_data')
def history_data():
    conn, cursor = None, None
    events = []

    try:
        conn = db_connect()
        cursor = conn.cursor(dictionary=True)

        # ------------------- AUDIO -------------------
        cursor.execute("""
            SELECT timestamp, rms, peak, audio_db, is_anomaly 
            FROM audio_data 
            ORDER BY timestamp DESC LIMIT 100
        """)
        for row in cursor.fetchall():
            events.append({
                "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if row["timestamp"] else "",
                "event_type": "Audio",
                "description": f"RMS: {row['rms']}, Peak: {row['peak']}, dB: {row['audio_db']}",
                "severity": "Danger" if row["is_anomaly"] else "Info"
            })

        # ------------------- ENVIRONMENTAL -------------------
        cursor.execute("""
            SELECT timestamp, temperature, humidity, gas, is_anomaly
            FROM environmental_data 
            ORDER BY timestamp DESC LIMIT 100
        """)
        for row in cursor.fetchall():
            events.append({
                "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if row["timestamp"] else "",
                "event_type": "Sensor",
                "description": f"Temp: {row['temperature']}°C, Humidity: {row['humidity']}%, Gas: {row['gas']}",
                "severity": "Danger" if row["is_anomaly"] else "Info"
            })

        # ------------------- VISION -------------------
        cursor.execute("""
            SELECT timestamp, behavior, is_anomaly
            FROM vision_data 
            ORDER BY timestamp DESC LIMIT 100
        """)
        for row in cursor.fetchall():
            events.append({
                "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if row["timestamp"] else "",
                "event_type": "Vision",
                "description": f"Behavior: {row['behavior']}",
                "severity": "Danger" if row["is_anomaly"] else "Info"
            })

        # ------------------- FUSION -------------------
        cursor.execute("""
            SELECT timestamp, fusion_label, fusion_confidence
            FROM fusion_predictions 
            ORDER BY timestamp DESC LIMIT 100
        """)
        for row in cursor.fetchall():
            events.append({
                "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if row["timestamp"] else "",
                "event_type": "Fusion",
                "description": f"Label: {row['fusion_label']}, Confidence: {row['fusion_confidence']:.2f}",
                "severity": "Warning" if row["fusion_confidence"] > 0.7 else "Info"
            })

        # ------------------- TRACKING -------------------
        cursor.execute("""
            SELECT first_seen, tracked_id, recovered
            FROM tracked_chickens 
            ORDER BY first_seen DESC LIMIT 100
        """)
        for row in cursor.fetchall():
            events.append({
                "timestamp": row["first_seen"].strftime("%Y-%m-%d %H:%M:%S") if row["first_seen"] else "",
                "event_type": "Tracking",
                "description": f"Chicken ID: {row['tracked_id']}, Recovered: {'Yes' if row['recovered'] else 'No'}",
                "severity": "Warning" if not row["recovered"] else "Info"
            })

        # ------------------- SORT -------------------
        def parse_time(val):
            try:
                return datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
            except:
                return datetime.min

        events_sorted = sorted(events, key=lambda x: parse_time(x["timestamp"]), reverse=True)

        # ------------------- FIXED: DataTables expects "data" -------------------
        return jsonify({"data": events_sorted})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if cursor: cursor.close()
        if conn: conn.close()


if __name__ == "__main__":
    main()
