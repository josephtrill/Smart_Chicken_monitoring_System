import mysql.connector
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

DB_HOST = os.environ.get("MYSQL_HOST", "127.0.0.1")
DB_PORT = int(os.environ.get("MYSQL_PORT", 3306))
DB_USER = os.environ.get("MYSQL_USER", "root")
DB_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
DB_NAME = os.environ.get("MYSQL_DB", "research")

# NEW Thresholds (Ground Truth)
ENV_MSE_THRESHOLD = 600.0
AUDIO_MSE_THRESHOLD = 0.5
VISION_MSE_THRESHOLD = 200.0

def get_conn():
    return mysql.connector.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME
    )

def evaluate(table, mse_col, threshold, name):
    try:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute(f"SELECT {mse_col}, is_anomaly FROM {table} ORDER BY id DESC LIMIT 500")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            print(f"[{name}] No data found.")
            return

        y_true = [] # Based on NEW threshold
        y_pred = [] # Based on OLD stored value

        for r in rows:
            if r[mse_col] is None: continue
            
            # Ground Truth (New Threshold)
            # If mse > new_threshold, it IS an anomaly (1). Else Normal (0).
            val = float(r[mse_col])
            actual = 1 if val > threshold else 0
            y_true.append(actual)
            
            # Prediction (Stored in DB)
            y_pred.append(int(r['is_anomaly']))

        if not y_true:
            print(f"[{name}] No valid data.")
            return

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"\n=== {name} Model Evaluation ===")
        print(f"Comparing Stored Predictions (Old) vs Refined Thresholds (New/Ground Truth)")
        print(f"Threshold Used: {threshold}")
        print("-" * 30)
        print(f"Confusion Matrix:\n{cm}")
        print(f" [TN, FP]")
        print(f" [FN, TP]")
        print("-" * 30)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        if cm[0][1] > 0:
            print(f"NOTE: {cm[0][1]} False Positives detected (Old model flagged Normal as Anomaly).")
        if cm[1][0] > 0:
            print(f"NOTE: {cm[1][0]} False Negatives detected (Old model missed Real Anomaly).")

    except Exception as e:
        with open("evaluation_results.txt", "a", encoding="utf-8") as f:
            f.write(f"Error evaluating {name}: {e}\n")

if __name__ == "__main__":
    with open("evaluation_results.txt", "w", encoding="utf-8") as f:
        f.write("Generating Confusion Matrices for Defense...\n")
    
    # Redirect stdout to file is hard in script, so we'll just modify evaluate to write to file
    # Actually, let's just modify evaluate() to take a file handle or write to file
    pass

# Redefine evaluate to write to file
def evaluate(table, mse_col, threshold, name):
    try:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute(f"SELECT {mse_col}, is_anomaly FROM {table} ORDER BY id DESC LIMIT 500")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        with open("evaluation_results.txt", "a", encoding="utf-8") as f:
            if not rows:
                f.write(f"[{name}] No data found.\n")
                return

            y_true = [] 
            y_pred = [] 

            for r in rows:
                if r[mse_col] is None: continue
                val = float(r[mse_col])
                actual = 1 if val > threshold else 0
                y_true.append(actual)
                y_pred.append(int(r['is_anomaly']))

            if not y_true:
                f.write(f"[{name}] No valid data.\n")
                return

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            f.write(f"\n=== {name} Model Evaluation ===\n")
            f.write(f"Comparing Stored Predictions (Old) vs Refined Thresholds (New/Ground Truth)\n")
            f.write(f"Threshold Used: {threshold}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Confusion Matrix:\n{cm}\n")
            f.write(f" [TN, FP]\n")
            f.write(f" [FN, TP]\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy:  {acc:.4f}\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall:    {rec:.4f}\n")
            f.write(f"F1 Score:  {f1:.4f}\n")
            
            # Classification Report
            report = classification_report(y_true, y_pred, labels=[0, 1], target_names=['Normal', 'Anomaly'], zero_division=0)
            f.write("\nClassification Report:\n")
            f.write(report + "\n")
            
            if cm[0][1] > 0:
                f.write(f"NOTE: {cm[0][1]} False Positives detected (Old model flagged Normal as Anomaly).\n")
            if cm[1][0] > 0:
                f.write(f"NOTE: {cm[1][0]} False Negatives detected (Old model missed Real Anomaly).\n")

    except Exception as e:
        with open("evaluation_results.txt", "a", encoding="utf-8") as f:
            f.write(f"Error evaluating {name}: {e}\n")

def evaluate_fusion():
    name = "Fusion"
    try:
        conn = get_conn()
        cur = conn.cursor(dictionary=True)
        # Fetch scores and label
        cur.execute("SELECT sensor_score, audio_score, vision_score, fusion_label FROM fusion_predictions ORDER BY id DESC LIMIT 500")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        with open("evaluation_results.txt", "a", encoding="utf-8") as f:
            if not rows:
                f.write(f"[{name}] No data found.\n")
                return

            y_true = [] 
            y_pred = [] 

            for r in rows:
                # Ground Truth: High Stress ONLY if ALL 3 scores are above NEW thresholds
                s_anom = 1 if float(r['sensor_score'] or 0) > ENV_MSE_THRESHOLD else 0
                a_anom = 1 if float(r['audio_score'] or 0) > AUDIO_MSE_THRESHOLD else 0
                v_anom = 1 if float(r['vision_score'] or 0) > VISION_MSE_THRESHOLD else 0
                
                actual = 1 if (s_anom + a_anom + v_anom) == 3 else 0
                
                # Prediction: Old model said "High Stress"
                pred = 1 if r['fusion_label'] == 'High Stress' else 0
                
                y_true.append(actual)
                y_pred.append(pred)

            if not y_true:
                f.write(f"[{name}] No valid data.\n")
                return

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            f.write(f"\n=== {name} Model Evaluation ===\n")
            f.write(f"Comparing Stored Predictions (Old) vs Refined Logic (New/Ground Truth)\n")
            f.write(f"Logic: High Stress requires ALL 3 modalities > New Thresholds\n")
            f.write("-" * 30 + "\n")
            f.write(f"Confusion Matrix:\n{cm}\n")
            f.write(f" [TN, FP]\n")
            f.write(f" [FN, TP]\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy:  {acc:.4f}\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall:    {rec:.4f}\n")
            f.write(f"F1 Score:  {f1:.4f}\n")
            
            # Classification Report
            report = classification_report(y_true, y_pred, labels=[0, 1], target_names=['Normal', 'High Stress'], zero_division=0)
            f.write("\nClassification Report:\n")
            f.write(report + "\n")

            if cm[0][1] > 0:
                f.write(f"NOTE: {cm[0][1]} False Positives detected (Old model flagged High Stress incorrectly).\n")

    except Exception as e:
        with open("evaluation_results.txt", "a", encoding="utf-8") as f:
            f.write(f"Error evaluating {name}: {e}\n")

if __name__ == "__main__":
    with open("evaluation_results.txt", "w", encoding="utf-8") as f:
        f.write("Generating Confusion Matrices for Defense...\n")
    evaluate("environmental_data", "mse_score", ENV_MSE_THRESHOLD, "Environmental")
    evaluate("audio_data", "mse_score", AUDIO_MSE_THRESHOLD, "Audio")
    evaluate("vision_data", "mse_score", VISION_MSE_THRESHOLD, "Vision")
    evaluate_fusion()
