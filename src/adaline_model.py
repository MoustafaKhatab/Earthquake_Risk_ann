import pandas as pd
import numpy as np

# ==========================================
# 1. DATA LOADING FUNCTION
# ==========================================
def load_seismic_data(filepath):
    df = pd.read_csv(filepath)
    # Extract features and labels
    X = df[['latitude', 'longitude', 'distance_min', 'count_radius', 'avg_magnitude']].values
    y = df['label'].values
    return X, y

# ==========================================
# 2. MANUAL PREPROCESSING
# ==========================================
def preprocess_data(X, y, split_ratio=0.8):
    np.random.seed(10)
    indices = np.random.permutation(len(X))
    X_shuff, y_shuff = X[indices], y[indices]

    train_size = int(split_ratio * len(X))
    X_train, X_test = X_shuff[:train_size], X_shuff[train_size:]
    y_train, y_test = y_shuff[:train_size], y_shuff[train_size:]

    # Standardization
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1 

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled, y_train, y_test, mean, std

# ==========================================
# 3. ADALINE MODEL CLASS
# ==========================================
class Adaline:
    def __init__(self, learning_rate=0.0001, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.cost = [] 

    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        for i in range(self.epochs):
            # 1. Linear Output (Net Input) - NO Step Function here!
            output = np.dot(X, self.weights) + self.bias
            
            # 2. Errors (Target - Continuous Output)
            errors = (y - output)
            
            # 3. Update Weights (Gradient Descent)
            # Update = lr * X_transpose * errors
            self.weights += self.lr * X.T.dot(errors)
            self.bias += self.lr * errors.sum()
            
            # 4. Calculate Cost (SSE) to track learning
            cost = (errors**2).sum() / 2.0
            self.cost.append(cost)
            
            if (i + 1) % 10 == 0:
                print(f"Epoch {i+1}/{self.epochs} - Cost: {cost:.4f}")

    def predict(self, X):
        # Activation (Step Function) only at the very end
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0.0, 1, 0)

# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # Load
    X_raw, y_raw = load_seismic_data("data/turkey_training_set.csv")
    
    # Preprocess
    X_train, X_test, y_train, y_test, mean, std = preprocess_data(X_raw, y_raw)
    
    # Train ADALINE
    # Note: Using a smaller LR for ADALINE to ensure convergence
    model = Adaline(learning_rate=0.0001, epochs=100)
    print("Starting ADALINE Training (Gradient Descent)...")
    model.fit(X_train, y_train)
    
    # Predict and Evaluate
    y_pred = model.predict(X_test)
    accuracy = np.sum(y_pred == y_test) / len(y_test)
    
    print("\n--- ADALINE RESULTS ---")
    print(f"Final Weights: {model.weights}")
    print(f"Final Bias: {model.bias}")
    print(f"Final Accuracy: {accuracy * 100:.2f}%")

def save_adaline_metadata(model, mean, std, filename="data/adaline_metadata.csv"):
    """
    Saves the trained ADALINE weights, bias, and scaling constants.
    """
    # Create the structured data
    # We include the 'Cost' as well to document the model's final performance
    final_cost = model.cost[-1]

    data = {
        'Feature': ['latitude', 'longitude', 'distance_min', 'count_radius', 'avg_magnitude', 'BIAS'],
        'Weight': np.append(model.weights, model.bias),
        'Mean': np.append(mean, np.nan),
        'Std': np.append(std, np.nan)
    }

    df_meta = pd.DataFrame(data)

    # Optional: Add a comment row or a separate file for the final cost
    df_meta.to_csv(filename, index=False)
    print(f"\n--- ADALINE Metadata Saved ---")
    print(f"Weights and Scaling constants stored in: {filename}")
    print(f"Final Training Cost (SSE): {final_cost:.6f}")

    # Call the function
save_adaline_metadata(model, mean, std)