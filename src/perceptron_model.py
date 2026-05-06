import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================
DATA_PATH = "data/turkey_training_set.csv"
LEARNING_RATE = 0.01
EPOCHS = 1000
TRAIN_SPLIT = 0.8

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Select Features (X) and Target (y)
X = df[['latitude', 'longitude', 'distance_min', 'count_radius', 'avg_magnitude']].values
y = df['label'].values

# ==========================================
# 2. MANUAL PREPROCESSING
# ==========================================
def preprocess_data(X, y, split_ratio=0.8):
    np.random.seed(50)
    indices = np.random.permutation(len(X))
    X_shuff, y_shuff = X[indices], y[indices]

    train_size = int(split_ratio * len(X))
    X_train, X_test = X_shuff[:train_size], X_shuff[train_size:]
    y_train, y_test = y_shuff[:train_size], y_shuff[train_size:]

    mean = np.mean(X_train, axis=0)
    std  = np.std(X_train, axis=0)
    std[std == 0] = 1  # Prevent division by zero

    X_train_scaled = (X_train - mean) / std
    X_test_scaled  = (X_test  - mean) / std

    return X_train_scaled, X_test_scaled, y_train, y_test, mean, std

X_train, X_test, y_train, y_test, mean, std = preprocess_data(X, y, TRAIN_SPLIT)
print(f"Dataset Split: {len(X_train)} Train | {len(X_test)} Test")

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
# 4. PERCEPTRON MODEL
# ==========================================
class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr      = learning_rate
        self.epochs  = epochs
        self.weights = None
        self.bias    = None
        self.epoch_errors = []   # tracks error rate per epoch for learning curve

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias    = 0

        for epoch in range(self.epochs):
            misclassifications = 0
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted   = 1 if linear_output >= 0 else 0

                update = self.lr * (y[idx] - y_predicted)
                if update != 0:
                    self.weights += update * x_i
                    self.bias    += update
                    misclassifications += 1

            # --- Learning Curve: error rate per epoch ---
            train_preds = self.predict(X)
            epoch_error = 1 - np.mean(train_preds == y)
            self.epoch_errors.append(epoch_error)

            # Print every 200 epochs (not 10 — avoids 1000 lines of output)
            if (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch+1:>4}/{self.epochs} | "
                      f"Misclassifications: {misclassifications:>4} | "
                      f"Train Error Rate: {epoch_error*100:.2f}%")

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

# ==========================================
# 5. EXECUTION & EVALUATION
# ==========================================
model = Perceptron(learning_rate=LEARNING_RATE, epochs=EPOCHS)
print("\nStarting Perceptron Training...")
model.fit(X_train, y_train)

print("\n--- Training Results ---")
print(f"Final Weights : {model.weights}")
print(f"Final Bias    : {model.bias}")

y_pred     = model.predict(X_test)
accuracy   = np.sum(y_pred == y_test) / len(y_test)
error_rate = 1 - accuracy

print(f"\nFinal Test Accuracy  : {accuracy   * 100:.2f}%")
print(f"Final Test Error Rate: {error_rate * 100:.2f}%")

# Confusion Matrix
confusion_matrix_report(y_test, y_pred, model_name="Perceptron")

# Learning Curve Summary (first and last few epochs)
print("\n--- Learning Curve (Error Rate per Epoch) ---")
print(f"  Epoch   1 Error: {model.epoch_errors[0]  * 100:.2f}%")
print(f"  Epoch 200 Error: {model.epoch_errors[199]* 100:.2f}%")
print(f"  Epoch 400 Error: {model.epoch_errors[399]* 100:.2f}%")
print(f"  Epoch 600 Error: {model.epoch_errors[599]* 100:.2f}%")
print(f"  Epoch 800 Error: {model.epoch_errors[799]* 100:.2f}%")
print(f"  Epoch1000 Error: {model.epoch_errors[-1] * 100:.2f}%")

# ==========================================
# 6. SAVE MODEL PARAMETERS
# ==========================================
def save_model_params(model, mean, std, filename="data/perceptron_metadata.csv"):
    data = {
        'Feature': ['latitude', 'longitude', 'distance_min', 'count_radius', 'avg_magnitude', 'BIAS'],
        'Weight' : np.append(model.weights, model.bias),
        'Mean'   : np.append(mean, np.nan),
        'Std'    : np.append(std,  np.nan)
    }
    pd.DataFrame(data).to_csv(filename, index=False)
    print(f"\nModel parameters saved to {filename}")


# this is just a ploting point to show how the data is 

def plot_decision_boundary(model, X_scaled, y, mean, std, 
                            feat_x=3, feat_y=4,
                            fname_x="count_radius", fname_y="avg_magnitude"):
    """
    Plots the decision boundary of the Perceptron using 2 features.
    feat_x=3 → count_radius (index 3)
    feat_y=4 → avg_magnitude (index 4)
    """

    # --- Step 1: Get the two feature columns ---
    x_vals = X_scaled[:, feat_x]
    y_vals = X_scaled[:, feat_y]

    # --- Step 2: Create the plot ---
    plt.figure(figsize=(9, 6))

    # Plot Safe points (label=0) in green
    plt.scatter(x_vals[y == 0], y_vals[y == 0],
                color='green', label='Safe (0)', alpha=0.5, s=30, edgecolors='white', linewidth=0.3)

    # Plot Risky points (label=1) in red
    plt.scatter(x_vals[y == 1], y_vals[y == 1],
                color='red', label='Risky (1)', alpha=0.7, s=30, edgecolors='white', linewidth=0.3)

    # --- Step 3: Draw the decision boundary line ---
    # The Perceptron decision boundary is where: w·x + b = 0
    # We solve for feat_y given feat_x:
    # w[feat_x]*x + w[feat_y]*y + b = 0
    # → y = -(w[feat_x]*x + b) / w[feat_y]

    w = model.weights
    b = model.bias

    x_line = np.linspace(x_vals.min() - 0.5, x_vals.max() + 0.5, 200)

    if w[feat_y] != 0:
        y_line = -(w[feat_x] * x_line + b) / w[feat_y]
        plt.plot(x_line, y_line, 'b-', linewidth=2, label='Decision Boundary')
    else:
        plt.axvline(x=-b / w[feat_x], color='blue', linewidth=2, label='Decision Boundary')

    # --- Step 4: Labels and styling ---
    plt.xlabel(f"{fname_x} (standardized)", fontsize=11)
    plt.ylabel(f"{fname_y} (standardized)", fontsize=11)
    plt.title("Perceptron Decision Boundary\n(count_radius vs avg_magnitude)", fontsize=13)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("decision_boundary.png", dpi=150)
    plt.show()
    print("Plot saved as decision_boundary.png")
save_model_params(model, mean, std)
plot_decision_boundary(model, X_train, y_train, mean, std)