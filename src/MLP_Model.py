import pandas as pd
import numpy as np

# ==========================================
# 1. FAULT DISTANCE FEATURE
# ==========================================
FAULT_CENTERS = [
    (40.77, 29.51),   
    (40.75, 31.61),   
    (39.77, 39.91),   
    (38.42, 38.68),   
    (37.57, 36.12),   
]

def haversine_min(lats, lons):
    
    R = 6371.0
    min_dists = np.full(len(lats), np.inf)

    for fc_lat, fc_lon in FAULT_CENTERS:
        lat1 = np.radians(lats)
        lon1 = np.radians(lons)
        lat2 = np.radians(fc_lat)
        lon2 = np.radians(fc_lon)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a    = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        dist = R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        min_dists = np.minimum(min_dists, dist)

    return min_dists


def add_fault_distance(X_raw):
    
    lats      = X_raw[:, 0]   
    lons      = X_raw[:, 1]   
    fault_dist = haversine_min(lats, lons).reshape(-1, 1)
    return np.hstack([X_raw, fault_dist])


# ==========================================
# 2. ACTIVATION FUNCTIONS
# ==========================================
def sigmoid(z):
    # Clipped for numerical stability
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    return a * (1 - a)


# ==========================================
# 3. MLP CLASS
# ==========================================
class EarthquakeMLP:
    def __init__(self, input_size=6, hidden_size=8, output_size=1, lr=0.1):
        np.random.seed(42)

        # Xavier initialization
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(1 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        self.lr          = lr
        self.train_costs = []
        self.val_costs   = []

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=6800):
        y_train = y_train.reshape(-1, 1)
        m       = X_train.shape[0]

        if y_val is not None:
            y_val = y_val.reshape(-1, 1)

        for i in range(epochs):
            # --- FORWARD PASS ---
            z1 = np.dot(X_train, self.w1) + self.b1
            a1 = sigmoid(z1)
            z2 = np.dot(a1, self.w2) + self.b2
            a2 = sigmoid(z2)

            # --- COST (MSE) ---
            error = y_train - a2
            cost  = np.mean(np.square(error))
            self.train_costs.append(cost)

            # --- VALIDATION COST ---
            if X_val is not None:
                _, a2_val = self.predict(X_val)
                val_cost  = np.mean(np.square(y_val - a2_val))
                self.val_costs.append(val_cost)

            # --- BACKPROPAGATION ---
            d_a2 = error * sigmoid_derivative(a2)
            d_a1 = d_a2.dot(self.w2.T) * sigmoid_derivative(a1)

            # --- WEIGHT UPDATES (divided by m) ---
            self.w2 += (a1.T.dot(d_a2) / m) * self.lr
            self.b2 += (np.sum(d_a2, axis=0, keepdims=True) / m) * self.lr
            self.w1 += (X_train.T.dot(d_a1) / m) * self.lr
            self.b1 += (np.sum(d_a1, axis=0, keepdims=True) / m) * self.lr

            # --- LOGGING every 200 epochs ---
            if (i + 1) % 200 == 0:
                train_acc = self._accuracy(X_train, y_train)
                log = (f"Epoch {i+1:>4}/{epochs} | "
                       f"Train MSE: {cost:.6f} | "
                       f"Train Acc: {train_acc*100:.2f}%")
                if X_val is not None:
                    val_acc = self._accuracy(X_val, y_val)
                    log += f" | Val MSE: {val_cost:.6f} | Val Acc: {val_acc*100:.2f}%"
                print(log)

    def _accuracy(self, X, y):
        preds, _ = self.predict(X)
        return np.mean(preds.flatten() == y.flatten())

    def predict(self, X):
        a1 = sigmoid(np.dot(X, self.w1) + self.b1)
        a2 = sigmoid(np.dot(a1, self.w2) + self.b2)
        return np.where(a2 >= 0.5, 1, 0), a2


# ==========================================
# 4. CONFUSION MATRIX
# ==========================================
def confusion_matrix_report(y_true, y_pred):
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy   = (TP + TN) / (TP + TN + FP + FN)
    error_rate = 1 - accuracy
    precision  = TP / (TP + FP + 1e-8)
    recall     = TP / (TP + FN + 1e-8)
    f1         = 2 * precision * recall / (precision + recall + 1e-8)

    print("\n" + "="*45)
    print("          CONFUSION MATRIX")
    print("="*45)
    print(f"  True  Positive (TP): {TP:>5}  ← Correctly predicted Risky")
    print(f"  True  Negative (TN): {TN:>5}  ← Correctly predicted Safe")
    print(f"  False Positive (FP): {FP:>5}  ← Predicted Risky, was Safe")
    print(f"  False Negative (FN): {FN:>5}  ← Predicted Safe,  was Risky ⚠️")
    print("-"*45)
    print(f"  Accuracy   : {accuracy   * 100:.2f}%")
    print(f"  Error Rate : {error_rate * 100:.2f}%")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1-Score   : {f1:.4f}")
    print("="*45)


# ==========================================
# 5. SAVE METADATA
# ==========================================
def save_mlp_metadata(model, mean, std, features, filename="data/mlp_metadata.csv"):
    rows = []
    for i, feat in enumerate(features):
        for j in range(model.w1.shape[1]):
            rows.append({'Type': 'W1', 'Source': feat, 'Target': f'H{j}', 'Value': model.w1[i, j]})
    for j in range(model.b1.shape[1]):
        rows.append({'Type': 'B1', 'Source': 'Bias', 'Target': f'H{j}', 'Value': model.b1[0, j]})
    for j in range(model.w2.shape[0]):
        rows.append({'Type': 'W2', 'Source': f'H{j}', 'Target': 'Output', 'Value': model.w2[j, 0]})
    rows.append({'Type': 'B2', 'Source': 'Bias', 'Target': 'Output', 'Value': model.b2[0, 0]})
    for i, feat in enumerate(features):
        rows.append({'Type': 'Scaling_Mean', 'Source': feat, 'Target': 'None', 'Value': mean[i]})
        rows.append({'Type': 'Scaling_Std',  'Source': feat, 'Target': 'None', 'Value': std[i]})

    pd.DataFrame(rows).to_csv(filename, index=False)
    print(f"\n--- [Phase 3] Metadata saved to {filename} ---")


# ==========================================
# 6. VISUALIZE PREDICTIONS
# ==========================================
def test_and_visualize(model, X_raw, X_scaled, y_true):
    preds, probs = model.predict(X_scaled)
    preds  = preds.flatten()
    probs  = probs.flatten()
    y_true = y_true.flatten()

    print("\n" + "="*80)
    print(f"{'LATITUDE':<10} | {'LONGITUDE':<10} | {'FAULT KM':<10} | "
          f"{'ACTUAL':<8} | {'PRED':<8} | {'PROB %'}")
    print("-"*80)

    n_show = min(200, len(y_true))
    for i in range(n_show):
        lat       = X_raw[i, 0]
        lon       = X_raw[i, 1]
        fault_km  = X_raw[i, 5]          # العمود الجديد
        act_str   = "Risky" if y_true[i] == 1 else "Safe"
        pre_str   = "Risky" if preds[i]  == 1 else "Safe"
        marker    = "✅" if y_true[i] == preds[i] else "❌"
        print(f"{lat:<10.4f} | {lon:<10.4f} | {fault_km:<10.1f} | "
              f"{act_str:<8} | {pre_str:<8} | {probs[i]*100:>6.2f}% {marker}")

    print("="*80)


# ==========================================
# 7. MAIN PIPELINE
# ==========================================
def run_final_mlp_project():

    base_features = ['latitude', 'longitude', 'distance_min',
                     'count_radius', 'avg_magnitude']
    all_features  = base_features + ['fault_distance']

    try:
        df = pd.read_csv("data/turkey_training_set.csv")
    except FileNotFoundError:
        print("Error: data/turkey_training_set.csv not found!")
        return

    X_raw = df[base_features].values
    y     = df['label'].values

    X_raw = add_fault_distance(X_raw)
    print(f"✓ fault_distance feature added — shape now: {X_raw.shape}")
    print(f"  Sample fault distances (km): {X_raw[:3, 5].round(1)}")

    # --- Shuffle & Split 80/20 ---
    np.random.seed(42)
    indices              = np.random.permutation(len(X_raw))
    X_shuff, y_shuff     = X_raw[indices], y[indices]

    split_idx            = int(0.8 * len(X_raw))
    X_train_raw          = X_shuff[:split_idx]
    X_test_raw           = X_shuff[split_idx:]
    y_train              = y_shuff[:split_idx]
    y_test               = y_shuff[split_idx:]

    # --- Standardization (training stats only — no data leakage) ---
    mean              = X_train_raw.mean(axis=0)
    std               = X_train_raw.std(axis=0)
    std[std == 0]     = 1                           # zero-std guard

    X_train_scaled    = (X_train_raw - mean) / std
    X_test_scaled     = (X_test_raw  - mean) / std

    # --- Train ---
    print("\n--- Phase 3: MLP (6-8-1 Architecture with fault_distance) ---\n")
    model = EarthquakeMLP(input_size=6, hidden_size=8, lr=0.1)
    model.fit(
        X_train_scaled, y_train,
        X_val=X_test_scaled, y_val=y_test,
        epochs=6800
    )

    # --- Results ---
    val_preds, _ = model.predict(X_test_scaled)
    val_acc      = np.mean(val_preds.flatten() == y_test)
    val_err      = 1 - val_acc

    print(f"\nFINAL VALIDATION ACCURACY : {val_acc * 100:.2f}%")
    print(f"FINAL VALIDATION ERROR    : {val_err * 100:.2f}%")

    confusion_matrix_report(y_test, val_preds.flatten())
    save_mlp_metadata(model, mean, std, all_features)
    test_and_visualize(model, X_test_raw, X_test_scaled, y_test)


if __name__ == "__main__":
    run_final_mlp_project()