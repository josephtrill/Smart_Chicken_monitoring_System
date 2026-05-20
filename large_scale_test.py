import random
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# --- CONFIGURATION ---
N_SAMPLES = 1000

# New Thresholds (The ones we are testing)
THRESHOLDS = {
    'env': 600.0,
    'audio': 0.5,
    'vision': 200.0
}

def generate_noisy_value(true_state, threshold, noise_level=0.05):
    """
    Generates a sensor value based on the true state (0=Normal, 1=Stress).
    Adds noise to simulate real-world imperfections (overlap).
    """
    if true_state == 0: # Normal
        # Mostly below threshold
        if random.random() < noise_level:
            # Simulate noise spike (False Positive scenario)
            return random.uniform(threshold * 1.01, threshold * 1.2)
        else:
            # Normal range
            return random.uniform(0, threshold * 0.95)
    else: # Stress
        # Mostly above threshold
        if random.random() < noise_level:
            # Simulate sensor failure/miss (False Negative scenario)
            return random.uniform(threshold * 0.8, threshold * 0.99)
        else:
            # Stress range
            return random.uniform(threshold * 1.05, threshold * 2.0)

def evaluate_model(name, threshold):
    y_true = [] # Ground Truth State
    y_pred = [] # Model Prediction
    
    for _ in range(N_SAMPLES):
        # 1. Generate Ground Truth
        # 80% Normal, 20% Stress
        state = 1 if random.random() < 0.2 else 0
        
        # 2. Generate Sensor Value (with noise)
        val = generate_noisy_value(state, threshold)
        
        # 3. Model Prediction
        pred = 1 if val > threshold else 0
        
        y_true.append(state)
        y_pred.append(pred)
        
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    with open("test_results.txt", "a", encoding="utf-8") as f:
        f.write(f"\n=== {name} Model Evaluation (New Threshold: {threshold}) ===\n")
        f.write(f"N={N_SAMPLES} (Synthetic Data with 5% Noise)\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
        f.write(f" [TN, FP]\n")
        f.write(f" [FN, TP]\n")
        
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly'], zero_division=0) + "\n")

def evaluate_fusion():
    y_true = []
    y_pred = []
    
    for _ in range(N_SAMPLES):
        # 1. Generate Ground Truth
        # 80% Normal, 20% High Stress
        state = 1 if random.random() < 0.2 else 0
        
        # 2. Generate Sensor Values (All 3 modalities)
        # Fusion Logic: High Stress means ALL 3 are stressed
        # So if state=1, we generate all 3 as stress (with independent noise)
        # If state=0, we generate at least one as normal
        
        if state == 1:
            s_val = generate_noisy_value(1, THRESHOLDS['env'])
            a_val = generate_noisy_value(1, THRESHOLDS['audio'])
            v_val = generate_noisy_value(1, THRESHOLDS['vision'])
        else:
            # Randomly pick which ones are normal/stress, but ensure NOT ALL 3 are stress
            # Simplification: Generate all as Normal for true normal, 
            # or mix them. Let's make it realistic:
            # Most normals have all normal. Some have 1 or 2 anomalies (Multi-Modal but not High Stress).
            
            rand = random.random()
            if rand < 0.7: # All Normal
                s_val = generate_noisy_value(0, THRESHOLDS['env'])
                a_val = generate_noisy_value(0, THRESHOLDS['audio'])
                v_val = generate_noisy_value(0, THRESHOLDS['vision'])
            elif rand < 0.9: # 1 Anomaly (e.g. Vision glitch)
                s_val = generate_noisy_value(0, THRESHOLDS['env'])
                a_val = generate_noisy_value(0, THRESHOLDS['audio'])
                v_val = generate_noisy_value(1, THRESHOLDS['vision'])
            else: # 2 Anomalies (Multi-Modal)
                s_val = generate_noisy_value(1, THRESHOLDS['env'])
                a_val = generate_noisy_value(1, THRESHOLDS['audio'])
                v_val = generate_noisy_value(0, THRESHOLDS['vision'])
        
        # 3. Model Prediction
        s_pred = 1 if s_val > THRESHOLDS['env'] else 0
        a_pred = 1 if a_val > THRESHOLDS['audio'] else 0
        v_pred = 1 if v_val > THRESHOLDS['vision'] else 0
        
        # Strict Fusion: All 3 must be 1
        final_pred = 1 if (s_pred + a_pred + v_pred) == 3 else 0
        
        y_true.append(state)
        y_pred.append(final_pred)
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    with open("test_results.txt", "a", encoding="utf-8") as f:
        f.write(f"\n=== Fusion Model Evaluation ===\n")
        f.write(f"N={N_SAMPLES} (Synthetic Data with Noise & Mixed States)\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")
        f.write(f" [TN, FP]\n")
        f.write(f" [FN, TP]\n")
        
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=['Normal', 'High Stress'], zero_division=0) + "\n")

if __name__ == "__main__":
    with open("test_results.txt", "w", encoding="utf-8") as f: f.write("")
    evaluate_model("Environmental", THRESHOLDS['env'])
    evaluate_model("Audio", THRESHOLDS['audio'])
    evaluate_model("Vision", THRESHOLDS['vision'])
    evaluate_fusion()
