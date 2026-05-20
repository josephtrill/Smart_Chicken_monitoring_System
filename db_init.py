import mysql.connector
import os

DB_HOST = os.environ.get("MYSQL_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("MYSQL_PORT", 3306))
DB_USER = os.environ.get("MYSQL_USER", "root")
DB_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
DB_NAME = os.environ.get("MYSQL_DB", "research")

def init_db():
    conn = mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cur = conn.cursor()
    
    # Create DB if not exists
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    cur.execute(f"USE {DB_NAME}")
    
    tables = [
        """CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) NOT NULL,
            email VARCHAR(255) NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            farm_name VARCHAR(255),
            location VARCHAR(255),
            role VARCHAR(50) DEFAULT 'user',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS environmental_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            device_id VARCHAR(255),
            timestamp DATETIME,
            temperature FLOAT,
            humidity FLOAT,
            gas FLOAT,
            mse_score FLOAT,
            is_anomaly INT
        )""",
        """CREATE TABLE IF NOT EXISTS audio_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            timestamp DATETIME,
            rms FLOAT,
            peak FLOAT,
            audio_db FLOAT,
            mse_score FLOAT
        )""",
        """CREATE TABLE IF NOT EXISTS vision_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            timestamp DATETIME,
            mse_score FLOAT,
            is_anomaly INT,
            frame_path VARCHAR(255)
        )""",
        """CREATE TABLE IF NOT EXISTS fusion_predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            timestamp DATETIME,
            sensor_anomaly INT,
            audio_anomaly INT,
            vision_anomaly INT,
            sensor_score FLOAT,
            audio_score FLOAT,
            vision_score FLOAT,
            fusion_label VARCHAR(255),
            fusion_confidence FLOAT
        )""",
        """CREATE TABLE IF NOT EXISTS alerts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            type VARCHAR(50),
            message TEXT,
            severity VARCHAR(50),
            timestamp DATETIME,
            resolved INT DEFAULT 0
        )""",
        """CREATE TABLE IF NOT EXISTS tracked_chickens (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            tracked_id VARCHAR(255),
            frame_path VARCHAR(255),
            vision_bbox VARCHAR(255),
            mse_score FLOAT,
            first_seen DATETIME,
            last_seen DATETIME,
            recovered INT DEFAULT 0,
            recovery_timestamp DATETIME,
            related_fusion_id INT
        )""",
        """CREATE TABLE IF NOT EXISTS egg_production (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            date DATE,
            total_eggs INT,
            broken_eggs INT,
            avg_stress_label VARCHAR(255),
            notes TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS devices (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT,
            device_uid VARCHAR(255) UNIQUE,
            device_type VARCHAR(255),
            location VARCHAR(255),
            status VARCHAR(50),
            last_seen DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )"""
    ]
    
    for table_sql in tables:
        try:
            cur.execute(table_sql)
            print(f"Table created/verified.")
        except Exception as e:
            print(f"Error creating table: {e}")
            
    conn.commit()
    cur.close()
    conn.close()
    print("Database initialized.")

if __name__ == "__main__":
    init_db()
