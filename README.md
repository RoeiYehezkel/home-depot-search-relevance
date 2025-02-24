# home-depot-search-relevance
---

## Preface
This task focuses on predicting the **relevance of a search phrase** to a corresponding product using the **Home Depot Product Search Relevance Competition** dataset. 

The dataset includes **search phrases, product titles, descriptions, and attributes**, along with **relevance scores** ranging from **1 to 3** (higher scores indicate a better match). The goal is to develop models capable of accurately predicting these scores to improve search rankings.

---

## Methodologies Explored
### 1. Baseline Models
- **Ridge Regression** and **CountVectorizer** establish benchmark performance.
- Simple and interpretable methods with low computational cost.

### 2. Deep Learning with Siamese Networks
- **Siamese Networks** compare search phrases with product data.
- Learn relationships between inputs via **shared feature space encoding** and similarity measurement.

### 3. Evaluation Metrics
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Error (MAE)**

### 4. Iterative Improvements
- **Hybrid Approaches**: Deep learning for feature extraction + machine learning (Random Forest, XGBoost) for predictions.

---

## Character-Level LSTM Model
### 1. Preprocessing
#### 1.1 Cleaning Text
- **Lowercasing** for consistency.
- **Whitespace normalization** to trim unnecessary spaces.

#### 1.2 Tokenization
- **Character-level tokenization** maps characters to unique numerical indices.
- Example: "hammer" → `[19, 12, 24, 24, 16, 29]`

#### 1.3 Sequence Padding
- Ensures uniform length across text fields.
- Example: `search_terms (max=100)`, `titles (max=70)`, `descriptions (max=500)`, `attributes (max=250)`.

### 2. Dataset Creation
- Custom **PyTorch Dataset** for character sequences and relevance scores.
- Dataset splits:
  - **Training (80%)**
  - **Validation (20%)**

### 3. Model Definition
- **Embedding Layer**: Maps character indices to dense vectors.
- **Bidirectional LSTM**:
  - **Hidden Size**: 64
  - **Layers**: 3
  - **Dropout**: 0.3
  - **Bidirectionality**: Enabled
- **Feature Extraction**:
  - Extract embeddings from last hidden state of LSTM.
  - Compute **absolute difference** between query and product embeddings.
- **Fully Connected Layer**: Processes extracted features for final relevance score.

### 4. Training
- **Loss Function**: SmoothL1Loss
- **Optimizer**: AdamW (LR: `1e-3`, weight decay: `1e-5`)
- **Scheduler**: ReduceLROnPlateau (adaptive LR based on validation loss)

---

## Random Forest & XGBoost on Extracted Features
### 1. Feature Extraction
- Extract **high-dimensional embeddings** from LSTM model.
- Pass features to traditional machine learning models.

### 2. Random Forest Training
- **n_estimators=100**
- **max_depth=6**
- **Random Seed=42**

### 3. XGBoost Training
- **n_estimators=100**
- **max_depth=6**
- **learning_rate=0.1**

---

## Ridge Regression Benchmark
- **CountVectorizer**: Character-level **n-gram representation**.
- **Ridge Regression** with **L2 regularization**.
- Hyperparameter: `alpha=1.0`.

---

## Results Comparison
| Model Type | Train RMSE | Val RMSE | Test RMSE | Train MAE | Val MAE | Test MAE |
|------------|------------|------------|------------|------------|------------|------------|
| **Baseline Ridge Regression** | 0.5050 | 0.5280 | 0.5428 | 0.4133 | 0.4313 | 0.4437 |
| **Character-Level LSTM** | 0.4545 | 0.4836 | 0.5454 | 0.3657 | 0.3862 | 0.4316 |
| **LSTM Features + Random Forest** | 0.4349 | 0.4836 | 0.5422 | 0.3497 | 0.3874 | 0.4350 |
| **LSTM Features + XGBoost** | 0.4267 | 0.4826 | 0.5432 | 0.3425 | 0.3861 | 0.4353 |

---

## Word-Level Siamese Convolutional Model
### 1. Preprocessing
- **Lowercasing, removing punctuation, whitespace normalization**.
- **Word-level tokenization** preserves meaningful phrases (e.g., "3/4 inch" → `['3/4', 'inch']`).

### 2. Dataset Creation
- **PyTorch Dataset** handles tokenized sequences & relevance scores.
- Word embeddings trained with **Word2Vec**.

### 3. Model Definition
- **Embedding Layer**: Pre-trained Word2Vec (300-dimension embeddings).
- **Convolutional Layers**: Kernel sizes `[3,5,7]`, max-pooling, dropout (0.3).
- **Global Adaptive Pooling** for feature aggregation.
- **Fully Connected Layers**: Final prediction from extracted features.

### 4. Training and Evaluation
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam (LR: `5e-3`, weight decay: `1e-4`)
- **Scheduler**: ReduceLROnPlateau

---

## Final Results Comparison
| Model Type | Train RMSE | Val RMSE | Test RMSE | Train MAE | Val MAE | Test MAE |
|------------|------------|------------|------------|------------|------------|------------|
| **Word-Level CNN** | 0.5041 | 0.5021 | 0.5472 | 0.4115 | 0.4114 | 0.4497 |
| **SBERT + Word-Level CNN** | 0.4685 | 0.4846 | 0.5288 | 0.3781 | 0.3923 | 0.4281 |
| **Word-Level CNN + Random Forest** | 0.1841 | 0.4957 | 0.5271 | 0.1472 | 0.4013 | 0.4294 |
| **Word-Level CNN + XGBoost** | 0.4889 | 0.5004 | 0.5175 | 0.4011 | 0.4096 | 0.4242 |

---

## Conclusion
- **XGBoost on extracted features performed best**, balancing speed & accuracy.
- **SBERT-based CNN outperformed Word2Vec CNN**, capturing better semantic relationships.
- **LSTM + Feature-Based Models showed strong performance** for hybrid approaches.
- **Future work**: Explore **transformer-based** architectures and **attention mechanisms** for further improvements.

---


