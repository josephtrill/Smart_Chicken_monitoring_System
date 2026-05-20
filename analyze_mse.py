import mysql.connector
import os
import numpy as np

DB_HOST = os.environ.get("MYSQL_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("MYSQL_PORT", 3306))
DB_USER = os.environ.get("MYSQL_USER", "root")
DB_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
DB_NAME = os.environ.get("MYSQL_DB", "research")

def get_conn():
    return mysql.connector.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
    )
def analyze(table, mse_col="mse_score"):
    try:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)
        
        if table == "fusion_predictions":
            cur.execute(f"SELECT {mse_col}, fusion_label FROM {table} ORDER BY id DESC LIMIT 100")
            rows = cur.fetchall()
            # Map fusion_label to is_anomaly logic
            for r in rows:
                r['is_anomaly'] = 1 if r['fusion_label'] != 'Normal' else 0
        else:
            cur.execute(f"SELECT {mse_col}, is_anomaly FROM {table} ORDER BY id DESC LIMIT 100")
            rows = cur.fetchall()
            
        with open("results.txt", "a", encoding="utf-8") as f:
            f.write(f"--- {table} ---\n")
            if not rows:
                f.write("No data found.\n")
                return

            mses = [float(r[mse_col]) for r in rows if r[mse_col] is not None]
            normal_mses = [float(r[mse_col]) for r in rows if r[mse_col] is not None and r['is_anomaly'] == 0]
            anomaly_mses = [float(r[mse_col]) for r in rows if r[mse_col] is not None and r['is_anomaly'] == 1]

            f.write(f"Total Records: {len(rows)}\n")
            if normal_mses:
                avg = np.mean(normal_mses)
                std = np.std(normal_mses)
                max_val = np.max(normal_mses)
                p95 = np.percentile(normal_mses, 95)
                p99 = np.percentile(normal_mses, 99)
                
                f.write(f"Normal MSE (n={len(normal_mses)}):\n")
                f.write(f"  Mean: {avg:.4f}\n")
                f.write(f"  Std Dev: {std:.4f}\n")
                f.write(f"  Max: {max_val:.4f}\n")
                f.write(f"  95th Percentile: {p95:.4f}\n")
                f.write(f"  99th Percentile: {p99:.4f}\n")
                
                # Suggested threshold: Mean + 3*StdDev (standard statistical anomaly detection)
                suggested_std = avg + (3 * std)
                # Or simply slightly above the max seen in normal operation
                suggested_max = max_val * 1.1
                
                f.write(f"SUGGESTED THRESHOLD (Mean + 3*Std): {suggested_std:.4f}\n")
                f.write(f"SUGGESTED THRESHOLD (Max + 10%): {suggested_max:.4f}\n")
            else:
                f.write("No Normal Data to calibrate.\n")

            if anomaly_mses:
                f.write(f"Anomaly MSE (n={len(anomaly_mses)}): Avg={np.mean(anomaly_mses):.4f}, Min={np.min(anomaly_mses):.4f}\n")
            f.write("\n")

    except Exception as e:
        with open("results.txt", "a") as f:
            f.write(f"Error analyzing {table}: {e}\n")

if __name__ == "__main__":
    # Clear file
    with open("results.txt", "w") as f: f.write("")
    analyze("environmental_data")
    analyze("audio_data")
    analyze("vision_data")
    analyze("fusion_predictions", mse_col="fusion_confidence")
