# Earthquake_Risk_ann
This project builds an ANN-based seismic risk classifier. It analyzes historical earthquake data to label locations as Safe or Risky by learning patterns from past activity. It doesn’t predict future earthquakes, but identifies areas with characteristics similar to historically high-risk zones.

# task one was to clean and work with only turkiye data
data/turkey_training_set.csv has all the data for the past 100 yeays ago

# features
    latitude: a geographic coordinate that specifies the north-south position of a point on the surface of the Earth or another celestial body.
    longitude: is a geographic coordinate that specifies the east-west position of a point on the surface of the Earth
    distance_min: How close is the nearest epicenter?
    count_radius: How many earthquakes happened within 50km?
    avg_magnitude: What is the average strength of those nearby events?

# lable 
    (0,1) : it is out lable , 0 means it no effect from this earthquike and 1 means it is effecting or there is a noticed impact

# how we gonna use this data?
 - we use the lat and lon to know where the earthquike happend.
 - we used this points also to point the country we wanna work with 

# our Perceptron equation 
 - The Perceptron takes your features, multiplies them by Weights ($W$), adds a Bias ($b$), and then checks if the total is greater than zero:
 - z = (w1 . lat) + (w2 . lon) + (w3 . dist_min) + (w4 count) + (w5 . avg_mag) + b

 * Activation: If z = 0, output is 1 (Risky). Otherwise, output is 0 (Safe).

## 🛠️ Key Features & Engineering
### 1. Manual Data Preprocessing
To demonstrate a deep understanding of the ML pipeline, we avoided high-level libraries (like `scikit-learn`) for data preparation:
- **Shuffle & Split:** Randomly permuted the dataset to ensure a balanced 80/20 train-test split.
- **Manual Standardization:** Implemented Z-score normalization ($x_{std} = \frac{x - \mu}{\sigma}$) to handle feature scale imbalances between Latitude and Distance.
- **Data Leakage Protection:** Calculated Mean and Standard Deviation exclusively from the Training Set.



### 2. The Perceptron Model
A custom class implementing the classic binary classifier:
- **Weight Initialization:** Weights and Bias are initialized to zero.
- **Activation Function:** Unit Step Function ($f(z) = 1$ if $z \ge 0$, else $0$).
- **Learning Rule:** Iterative weight updates based on misclassifications:
  $$W_{new} = W_{old} + \eta \cdot (y_{target} - y_{predicted}) \cdot x$$

## 📊 Results: Phase 1 (Perceptron)
The Perceptron model achieved a near-perfect classification on the Turkey dataset.

- **Training Accuracy:** 100% (Converged by Epoch 10)
- **Test Accuracy:** 99.75%
- **Key Insight:** The `count_radius` feature showed the highest weight, indicating it is the most significant indicator for the risk threshold defined in the proposal.

## 📉 Phase 2: ADALINE Implementation (Adaptive Linear Neuron)

In this phase, the model was upgraded from the simple Perceptron rule to **ADALINE**, introducing **Batch Gradient Descent**. Unlike the Perceptron, ADALINE updates weights based on a continuous Cost Function (Sum of Squared Errors).

### ⚙️ Technical Specifications
- **Optimization:** Batch Gradient Descent.
- **Cost Function:** $J(w) = \frac{1}{2} \sum (y^{(i)} - \phi(z^{(i)}))^2$.
- **Learning Rate ($\eta$):** 0.0001.
- **Epochs:** 100.

### 📊 Comparative Results
| Metric | Perceptron (Phase 1) | ADALINE (Phase 2) |
| :--- | :--- | :--- |
| **Final Accuracy** | **99.75%** | **47.25%** |
| **Update Rule** | Error-based (Heaviside) | Gradient-based (Linear) |
| **Convergence** | Immediate (Epoch 10) | Stable (Cost decreased to 28.52) |

### 🔍 Analysis of "Accuracy Drop"
While ADALINE successfully minimized the **Cost (SSE)** from ~32.88 down to 28.52, the final classification accuracy was significantly lower than the Perceptron. 

**Observations:**
1. **Linear Separability:** The Perceptron excels at hard-labeled thresholds. ADALINE's linear activation attempts to minimize the distance to the target values (0 and 1), which, in this specific feature space, resulted in a decision boundary that misclassified a large portion of the test set.
2. **Convergence vs. Accuracy:** The model "converged" (the cost stopped moving), but it converged to a **Local Minimum** or a plane that does not suit binary classification for this specific dataset as well as the Perceptron's step-function approach.
3. **Transition to MLP:** This result highlights the limitations of linear models and justifies the move to **Phase 3: Multi-Layer Perceptron (MLP)**, where non-linear activation functions (Sigmoid/ReLU) and hidden layers will be used to resolve this classification conflict.
