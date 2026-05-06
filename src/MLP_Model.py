import pandas as pd
import numpy as np

# ==========================================
# 1. ACTIVATION FUNCTIONS
# ==========================================
def sigmoid(z):
    # Clipped for numerical stability — prevents exp overflow on large negative z
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(a):
    # 'a' is already the sigmoid output, so derivative is a*(1-a)
    return a * (1 - a)

# ==========================================
# 2. MLP CLASS
# ==========================================
class EarthquakeMLP:
    def __init__(self, input_size=5, hidden_size=8, output_size=1, lr=0.1):
        np.random.seed(42)

        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(1 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
        self.b2 = np.zeros((1, output_size))

        self.lr = lr
        self.train_costs = []
        self.val_costs   = []   # FIX 3: track validation cost each epoch

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=2000):
        y_train = y_train.reshape(-1, 1)
        m = X_train.shape[0]   # FIX 2: batch size for gradient normalization

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

            # --- FIX 3: Validation cost ---
            if X_val is not None:
                _, a2_val = self.predict(X_val)
                val_cost = np.mean(np.square(y_val - a2_val))
                self.val_costs.append(val_cost)

            # --- BACKPROPAGATION ---
            d_a2 = error * sigmoid_derivative(a2)
            d_a1 = d_a2.dot(self.w2.T) * sigmoid_derivative(a1)

            # --- FIX 2: Divide by m so gradient magnitude doesn't scale with dataset size ---
            # Without this, lr=0.1 on 1600 samples acts like lr=160 effectively
            self.w2 += (a1.T.dot(d_a2) / m) * self.lr
            self.b2 += (np.sum(d_a2, axis=0, keepdims=True) / m) * self.lr
            self.w1 += (X_train.T.dot(d_a1) / m) * self.lr
            self.b1 += (np.sum(d_a1, axis=0, keepdims=True) / m) * self.lr

            # --- Logging ---
            if (i + 1) % 200 == 0:
                train_acc = self._accuracy(X_train, y_train)
                log = f"Epoch {i+1:>4}/{epochs} | Train MSE: {cost:.6f} | Train Acc: {train_acc*100:.2f}%"
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
# 3. EVALUATION — FIX 5: Confusion Matrix
# ==========================================
def confusion_matrix_report(y_true, y_pred):
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (TP + TN) / (TP + TN + FP + FN)

    print("\n" + "="*40)
    print("       CONFUSION MATRIX")
    print("="*40)
    print(f"  True  Positive (TP): {TP:>5}")
    print(f"  True  Negative (TN): {TN:>5}")
    print(f"  False Positive (FP): {FP:>5}  ← predicted Risky, was Safe")
    print(f"  False Negative (FN): {FN:>5}  ← predicted Safe,  was Risky")
    print("-"*40)
    print(f"  Accuracy  : {accuracy*100:.2f}%")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print("="*40)

# ==========================================
# 4. UTILITY FUNCTIONS
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
    print(f"\n--- [Phase 3] Metadata Saved to {filename} ---")

def test_and_visualize(model, X_raw, X_scaled, y_true):
    preds, probs = model.predict(X_scaled)
    preds  = preds.flatten()
    probs  = probs.flatten()
    y_true = y_true.flatten()

    print("\n" + "="*70)
    print(f"{'LATITUDE':<10} | {'LONGITUDE':<10} | {'ACTUAL':<8} | {'PRED':<8} | {'PROB %'}")
    print("-"*70)

    # FIX 6: guard against test set smaller than 15
    n_show = min(200, len(y_true))
    for i in range(n_show):
        lat, lon  = X_raw[i, 0], X_raw[i, 1]
        act_str   = "Risky" if y_true[i] == 1 else "Safe"
        pre_str   = "Risky" if preds[i]  == 1 else "Safe"
        marker    = "✅" if y_true[i] == preds[i] else "❌"
        print(f"{lat:<10.4f} | {lon:<10.4f} | {act_str:<8} | {pre_str:<8} | {probs[i]*100:>6.2f}% {marker}")

    print("="*70)

# ==========================================
# 5. MAIN EXECUTION PIPELINE
# ==========================================
def run_final_mlp_project():
    features = ['latitude', 'longitude', 'distance_min', 'count_radius', 'avg_magnitude']

    try:
        df = pd.read_csv("data/turkey_training_set.csv")
    except FileNotFoundError:
        print("Error: data/turkey_training_set.csv not found!")
        return

    X = df[features].values
    y = df['label'].values

    # 80/20 Shuffle & Split
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    X_shuff, y_shuff = X[indices], y[indices]

    split_idx = int(0.8 * len(X))
    X_train_raw, X_test_raw = X_shuff[:split_idx],  X_shuff[split_idx:]
    y_train,     y_test     = y_shuff[:split_idx],  y_shuff[split_idx:]

    # Standardization — train stats only, with zero-std guard
    mean = X_train_raw.mean(axis=0)
    std  = X_train_raw.std(axis=0)
    std[std == 0] = 1                                    # FIX 4: prevent NaN

    X_train_scaled = (X_train_raw - mean) / std
    X_test_scaled  = (X_test_raw  - mean) / std

    # Train
    print("\n--- Phase 3: Multi-Layer Perceptron (5-8-1 Architecture) ---\n")
    model = EarthquakeMLP(input_size=5, hidden_size=8, lr=0.1)
    model.fit(X_train_scaled, y_train,
              X_val=X_test_scaled, y_val=y_test,
              epochs=6800)

    # Final results
    val_preds, _ = model.predict(X_test_scaled)
    val_acc = np.mean(val_preds.flatten() == y_test)
    print(f"\nFINAL VALIDATION ACCURACY (20% Unseen): {val_acc * 100:.2f}%")

    confusion_matrix_report(y_test, val_preds.flatten())
    save_mlp_metadata(model, mean, std, features)
    test_and_visualize(model, X_test_raw, X_test_scaled, y_test)
    

if __name__ == "__main__":
    run_final_mlp_project()