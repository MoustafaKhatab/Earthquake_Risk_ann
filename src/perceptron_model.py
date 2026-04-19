import pandas as pd
import numpy as np

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================
DATA_PATH = "data/turkey_training_set.csv"
LEARNING_RATE = 0.01
EPOCHS = 100
TRAIN_SPLIT = 0.8

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Select Features (X) and Target (y)
X = df[['latitude', 'longitude', 'distance_min', 'count_radius', 'avg_magnitude']].values
y = df['label'].values

# ==========================================
# 2. MANUAL PREPROCESSING (Week 2/3)
# ==========================================
def preprocess_data(X, y, split_ratio=0.8):
    # Shuffle indices to keep X and y synced
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    X_shuff, y_shuff = X[indices], y[indices]

    # Split into Train/Test
    train_size = int(split_ratio * len(X))
    X_train, X_test = X_shuff[:train_size], X_shuff[train_size:]
    y_train, y_test = y_shuff[:train_size], y_shuff[train_size:]

    # Manual Scaling (Standardization)
    # Use only training stats to avoid data leakage
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1  # Prevent division by zero

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_data(X, y, TRAIN_SPLIT)
print(f"Dataset Split: {len(X_train)} Train | {len(X_test)} Test")

# ==========================================
# 3. PERCEPTRON MODEL (Week 3)
# ==========================================
class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Initialize parameters
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        for epoch in range(self.epochs):
            errors = 0
            for idx, x_i in enumerate(X):
                # Linear combination + Activation (Step Function)
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = 1 if linear_output >= 0 else 0
                
                # Learning Rule: Update = lr * (target - prediction)
                update = self.lr * (y[idx] - y_predicted)
                
                if update != 0:
                    self.weights += update * x_i
                    self.bias += update
                    errors += 1
            
            # Optional: print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                 print(f"Epoch {epoch+1}/{self.epochs} - Misclassifications: {errors}")

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

# ==========================================
# 4. EXECUTION & EVALUATION
# ==========================================
# Initialize and Train
model = Perceptron(learning_rate=LEARNING_RATE, epochs=EPOCHS)
print("\nStarting Training...")
model.fit(X_train, y_train)

# Results
print("\n--- Training Results ---")
print(f"Final Weights: {model.weights}")
print(f"Final Bias: {model.bias}")

# Evaluation
y_pred = model.predict(X_test)
accuracy = np.sum(y_pred == y_test) / len(y_test)
print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")