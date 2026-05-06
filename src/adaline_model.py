import pandas as pd
import numpy as np

# ==========================================
# 1. DATA LOADING
# ==========================================
def load_seismic_data(filepath):
    df = pd.read_csv(filepath)
    X  = df[['latitude', 'longitude', 'distance_min', 'count_radius', 'avg_magnitude']].values
    y  = df['label'].values
    return X, y

# ==========================================
# 2. MANUAL PREPROCESSING
# ==========================================
def preprocess_data(X, y, split_ratio=0.8):
    np.random.seed(10)
    indices        = np.random.permutation(len(X))
    X_shuff, y_shuff = X[indices], y[indices]

    train_size = int(split_ratio * len(X))
    X_train, X_test = X_shuff[:train_size], X_shuff[train_size:]
    y_train, y_test = y_shuff[:train_size], y_shuff[train_size:]

    mean = np.mean(X_train, axis=0)
    std  = np.std(X_train,  axis=0)
    std[std == 0] = 1  # Prevent division by zero

    X_train_scaled = (X_train - mean) / std
    X_test_scaled  = (X_test  - mean) / std

    return X_train_scaled, X_test_scaled, y_train, y_test, mean, std

# ==========================================
# 3. CONFUSION MATRIX FUNCTION (shared)
# ==========================================
def confusion_matrix_report(y_true, y_pred, model_name="Model"):
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))

    accuracy   = (TP + TN) / (TP + TN + FP + FN)
    error_rate = 1 - accuracy
    precision  = TP / (TP + FP + 1e-8)
    recall     = TP / (TP + FN + 1e-8)
    f1         = 2 * precision * recall / (precision + recall + 1e-8)

    print(f"\n{'='*45}")
    print(f"   CONFUSION MATRIX — {model_name}")
    print(f"{'='*45}")
    print(f"  True  Positive (TP): {TP:>5}  ← Correctly predicted Risky")
    print(f"  True  Negative (TN): {TN:>5}  ← Correctly predicted Safe")
    print(f"  False Positive (FP): {FP:>5}  ← Predicted Risky, was Safe")
    print(f"  False Negative (FN): {FN:>5}  ← Predicted Safe,  was Risky ⚠️")
    print(f"{'-'*45}")
    print(f"  Accuracy   : {accuracy   * 100:.2f}%")
    print(f"  Error Rate : {error_rate * 100:.2f}%")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"{'='*45}")

# ==========================================
# 4. ADALINE MODEL
# ==========================================
class Adaline:
    def __init__(self, learning_rate=0.0001, epochs=1000):
        self.lr      = learning_rate
        self.epochs  = epochs
        self.weights = None
        self.bias    = None
        self.cost         = []   # SSE per epoch  (learning curve)
        self.epoch_errors = []   # error rate per epoch (learning curve)

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias    = 0

        for i in range(self.epochs):
            # Linear output — NO step function during training
            output = np.dot(X, self.weights) + self.bias
            errors = y - output

            # Batch gradient descent weight update
            self.weights += self.lr * X.T.dot(errors)
            self.bias    += self.lr * errors.sum()

            # SSE cost
            cost = (errors ** 2).sum() / 2.0
            self.cost.append(cost)

            # Error rate for learning curve
            train_preds = self.predict(X)
            epoch_error = 1 - np.mean(train_preds == y)
            self.epoch_errors.append(epoch_error)

            # Print every 200 epochs
            if (i + 1) % 200 == 0:
                print(f"Epoch {i+1:>4}/{self.epochs} | "
                      f"SSE Cost: {cost:.4f} | "
                      f"Train Error Rate: {epoch_error*100:.2f}%")

    def predict(self, X):
        # Step function only at prediction time
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0.0, 1, 0)

# ==========================================
# 5. SAVE METADATA
# ==========================================
def save_adaline_metadata(model, mean, std, filename="data/adaline_metadata.csv"):
    final_cost = model.cost[-1]
    data = {
        'Feature': ['latitude', 'longitude', 'distance_min', 'count_radius', 'avg_magnitude', 'BIAS'],
        'Weight' : np.append(model.weights, model.bias),
        'Mean'   : np.append(mean, np.nan),
        'Std'    : np.append(std,  np.nan)
    }
    pd.DataFrame(data).to_csv(filename, index=False)
    print(f"\n--- ADALINE Metadata Saved to {filename} ---")
    print(f"Final Training Cost (SSE): {final_cost:.6f}")

# ==========================================
# 6. EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # Load
    X_raw, y_raw = load_seismic_data("data/turkey_training_set.csv")

    # Preprocess
    X_train, X_test, y_train, y_test, mean, std = preprocess_data(X_raw, y_raw)
    print(f"Dataset Split: {len(X_train)} Train | {len(X_test)} Test")

    # Train
    model = Adaline(learning_rate=0.0001, epochs=1000)
    print("\nStarting ADALINE Training (Batch Gradient Descent)...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred     = model.predict(X_test)
    accuracy   = np.sum(y_pred == y_test) / len(y_test)
    error_rate = 1 - accuracy

    print(f"\n--- ADALINE RESULTS ---")
    print(f"Final Weights  : {model.weights}")
    print(f"Final Bias     : {model.bias}")
    print(f"Final Accuracy : {accuracy   * 100:.2f}%")
    print(f"Final Error Rate: {error_rate * 100:.2f}%")

    # Confusion Matrix
    confusion_matrix_report(y_test, y_pred, model_name="ADALINE")

    # Learning Curve Summary
    print("\n--- Learning Curve (SSE Cost per Epoch) ---")
    print(f"  Epoch   1 Cost: {model.cost[0]  :.4f}")
    print(f"  Epoch 200 Cost: {model.cost[199]:.4f}")
    print(f"  Epoch 400 Cost: {model.cost[399]:.4f}")
    print(f"  Epoch 600 Cost: {model.cost[599]:.4f}")
    print(f"  Epoch 800 Cost: {model.cost[799]:.4f}")
    print(f"  Epoch1000 Cost: {model.cost[-1] :.4f}")

    # Save
    save_adaline_metadata(model, mean, std)