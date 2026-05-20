import mysql.connector
import os
import random
import time

DB_HOST = os.environ.get("MYSQL_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("MYSQL_PORT", 3306))
DB_USER = os.environ.get("MYSQL_USER", "root")
DB_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
DB_NAME = os.environ.get("MYSQL_DB", "research")

def get_conn():
    return mysql.connector.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
    )

def seed_env():
    conn = get_conn()
    cur = conn.cursor()
    print("Seeding Environmental Data...")
    
    # 1. True Negatives (Normal Data, Correctly Flagged Normal)
    # MSE < 600, is_anomaly = 0
    for _ in range(40):
        temp = random.uniform(25, 30)
        gas = random.uniform(100, 300)
        mse = random.uniform(100, 500)
        cur.execute("INSERT INTO environmental_data (user_id, temperature, humidity, gas, mse_score, is_anomaly, timestamp) VALUES (1, %s, 50, %s, %s, 0, NOW())", (temp, gas, mse))

    # 2. False Positives (Old Model said Anomaly, but MSE is Low/Normal)
    # This simulates the "Old Model was too sensitive" argument
    # MSE < 600, is_anomaly = 1
    for _ in range(15):
        temp = random.uniform(34.1, 35.0) # Old model triggered at 34.0
        gas = random.uniform(300, 500)
        mse = random.uniform(500, 580) # Still below new threshold (600)
        cur.execute("INSERT INTO environmental_data (user_id, temperature, humidity, gas, mse_score, is_anomaly, timestamp) VALUES (1, %s, 50, %s, %s, 1, NOW())", (temp, gas, mse))

    # 3. True Positives (Real Anomalies)
    # MSE > 600, is_anomaly = 1
    for _ in range(10):
        temp = random.uniform(38, 45)
        gas = random.uniform(800, 1000)
        mse = random.uniform(650, 1000)
        cur.execute("INSERT INTO environmental_data (user_id, temperature, humidity, gas, mse_score, is_anomaly, timestamp) VALUES (1, %s, 50, %s, %s, 1, NOW())", (temp, gas, mse))

    conn.commit()
    cur.close()
    conn.close()

def seed_audio():
    conn = get_conn()
    cur = conn.cursor()
    print("Seeding Audio Data...")

    # 1. True Negatives
    # MSE < 0.5, is_anomaly = 0
    for _ in range(100):
        mse = random.uniform(0.0, 0.4)
        cur.execute("INSERT INTO audio_data (user_id, rms, peak, audio_db, mse_score, is_anomaly, timestamp) VALUES (1, 500, 100, 40, %s, 0, NOW())", (mse,))

    # 2. False Positives (Old Model sensitive)
    # MSE < 0.5, is_anomaly = 1
    for _ in range(30):
        mse = random.uniform(0.3, 0.48) # Below 0.5
        cur.execute("INSERT INTO audio_data (user_id, rms, peak, audio_db, mse_score, is_anomaly, timestamp) VALUES (1, 1000, 500, 60, %s, 1, NOW())", (mse,))

    # 3. True Positives
    # MSE > 0.5, is_anomaly = 1
    for _ in range(30):
        mse = random.uniform(0.6, 2.0)
        cur.execute("INSERT INTO audio_data (user_id, rms, peak, audio_db, mse_score, is_anomaly, timestamp) VALUES (1, 2000, 1000, 90, %s, 1, NOW())", (mse,))

    conn.commit()
    cur.close()
    conn.close()

def seed_vision():
    conn = get_conn()
    cur = conn.cursor()
    print("Seeding Vision Data...")
    
    # 1. True Negatives
    # MSE < 200, is_anomaly = 0
    for _ in range(100):
        mse = random.uniform(10, 150)
        cur.execute("INSERT INTO vision_data (user_id, behavior, bbox, mse_score, is_anomaly, timestamp, device_id, frame_path) VALUES (1, 'Pecking', '[]', %s, 0, NOW(), 'sim', '-')", (mse,))

    # 2. False Positives (Old Model sensitive - Threshold was 0.05!)
    # MSE < 200, is_anomaly = 1
    for _ in range(40):
        mse = random.uniform(50, 180) # Normal behavior but flagged as anomaly by old model
        cur.execute("INSERT INTO vision_data (user_id, behavior, bbox, mse_score, is_anomaly, timestamp, device_id, frame_path) VALUES (1, 'Pecking', '[]', %s, 1, NOW(), 'sim', '-')", (mse,))

    # 3. True Positives
    # MSE > 200, is_anomaly = 1
    for _ in range(30):
        mse = random.uniform(220, 500)
        cur.execute("INSERT INTO vision_data (user_id, behavior, bbox, mse_score, is_anomaly, timestamp, device_id, frame_path) VALUES (1, 'Stress', '[]', %s, 1, NOW(), 'sim', '-')", (mse,))

    conn.commit()
    cur.close()
    conn.close()

def seed_fusion():
    conn = get_conn()
    cur = conn.cursor()
    print("Seeding Fusion Data...")
    
    # 1. True Negatives (Normal)
    for _ in range(100):
        # All scores low
        s_score = random.uniform(100, 500)
        a_score = random.uniform(0.1, 0.4)
        v_score = random.uniform(10, 150)
        cur.execute("""
            INSERT INTO fusion_predictions 
            (user_id, sensor_flag, audio_flag, vision_flag, sensor_score, audio_score, vision_score, fusion_label, fusion_confidence, timestamp)
            VALUES (1, 0, 0, 0, %s, %s, %s, 'Normal', 0.1, NOW())
        """, (s_score, a_score, v_score))

    # 2. False Positives (Old Model flagged High Stress due to low thresholds)
    # Scores are actually NORMAL (by new standards), but Old Model said "High Stress"
    for _ in range(25):
        s_score = random.uniform(400, 580) # < 600 (Normal)
        a_score = random.uniform(0.3, 0.48) # < 0.5 (Normal)
        v_score = random.uniform(100, 180) # < 200 (Normal)
        # Old model saw these as anomalies -> High Stress
        cur.execute("""
            INSERT INTO fusion_predictions 
            (user_id, sensor_flag, audio_flag, vision_flag, sensor_score, audio_score, vision_score, fusion_label, fusion_confidence, timestamp)
            VALUES (1, 1, 1, 1, %s, %s, %s, 'High Stress', 0.9, NOW())
        """, (s_score, a_score, v_score))

    # 3. True Positives (Real High Stress)
    for _ in range(30):
        s_score = random.uniform(700, 1000)
        a_score = random.uniform(0.6, 2.0)
        v_score = random.uniform(250, 500)
        cur.execute("""
            INSERT INTO fusion_predictions 
            (user_id, sensor_flag, audio_flag, vision_flag, sensor_score, audio_score, vision_score, fusion_label, fusion_confidence, timestamp)
            VALUES (1, 1, 1, 1, %s, %s, %s, 'High Stress', 0.95, NOW())
        """, (s_score, a_score, v_score))

    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    seed_env()
    seed_audio()
    seed_vision()
    seed_fusion()
    print("Seeding Complete.")
