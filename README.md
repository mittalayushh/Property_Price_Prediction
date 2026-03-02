# 🏠 Ames Housing Price Prediction System  
## From Raw Real Estate Data to Production-Ready Machine Learning Pipeline  

---

## 📌 Project Overview

This project implements a complete **end-to-end Machine Learning regression system** to predict residential property prices using the Ames Housing dataset from OpenML.

The objective is to demonstrate a production-style ML workflow including:

- Data cleaning and preprocessing  
- Feature engineering and encoding  
- Outlier handling  
- Log transformation for stability  
- Pipeline-based model training  
- Cross-validation  
- Hyperparameter tuning  
- Model serialization for deployment  

The final model is an optimized **Random Forest Regressor** saved for future deployment in web applications or APIs.

---

## 📊 Dataset Information

- **Dataset:** Ames Housing Dataset  
- **Source:** OpenML (`house_prices`)  
- **Target Variable:** `SalePrice`  
- **Total Samples:** ~1460  
- **Problem Type:** Regression  

The dataset contains structured tabular features related to:

- Property size
- Location
- Quality
- Year built
- Interior & exterior characteristics

---

## 🧹 Data Preprocessing

### 1️⃣ Handling Missing Values
- Dropped high-missing-value columns:
  - `Alley`
  - `PoolQC`
  - `Fence`
  - `MiscFeature`
  - `FireplaceQu`
- Filled essential missing values:
  - `LotFrontage` → Median
  - `MasVnrArea` → 0

---

### 2️⃣ Feature Encoding Techniques

| Feature | Encoding Method |
|----------|----------------|
| MSZoning | One-Hot Encoding |
| ExterQual | Ordinal Encoding |
| KitchenQual | Ordinal Encoding |
| CentralAir | Binary Encoding |
| Neighborhood | Frequency Encoding |

---

### 3️⃣ Outlier Handling

- Clipped `GrLivArea` between 1st and 99th percentiles.
- Created `GrLivArea_clipped`.

---

### 4️⃣ Log Transformation

To reduce skewness and stabilize variance:

- `log_size = log1p(GrLivArea_clipped)`
- `log_price = log1p(SalePrice)`

Model training was performed on `log_price`.

---

## ⚙️ Model Architecture

### Train-Test Split
- 80% Training
- 20% Testing
- Random State: 42

---

### ML Pipeline Structure


ColumnTransformer
→ SimpleImputer (Median)
→ StandardScaler
RandomForestRegressor


**Model Parameters:**
- `n_estimators = 100`
- `random_state = 42`
- `n_jobs = -1`

Pipeline ensures:
- Clean preprocessing
- No data leakage
- Reproducibility

---

## 📈 Model Evaluation

Evaluation performed on original price scale (after reversing log transformation).

Metrics used:

- **MAE (Mean Absolute Error)**
- **RMSE (Root Mean Squared Error)**
- **R² Score**

Additionally:

- 5-Fold Cross Validation (R² scoring)

This ensures the model generalizes well across unseen data.

---

## 🔍 Hyperparameter Tuning

Used:


RandomizedSearchCV


Parameters tuned:
- `n_estimators`
- `max_depth`
- `max_features`

Best model saved as:


ames_housing_model.joblib


---

## 📊 Visualizations

The project includes:

- Actual vs Predicted Price Scatter Plot  
- Residual Distribution Histogram  
- Feature Importance Bar Chart  

These help evaluate prediction behavior and model interpretability.

---

## 💾 Model Persistence

Saved artifacts:

| File | Description |
|------|------------|
| `ames_housing_model.joblib` | Best tuned pipeline |
| `logistic_model.pkl` | Random Forest model |
| `feature_columns.pkl` | Feature column order |

Saved using:


joblib.dump()


These files allow deployment without retraining.

---

## 🛠️ Technology Stack

| Component | Technology |
|------------|------------|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn |
| Model | RandomForestRegressor |
| Hyperparameter Tuning | RandomizedSearchCV |
| Dataset Source | OpenML |

---

## 🚀 How to Run the Project

### 1️⃣ Clone Repository

git clone <repository-link>
cd <project-folder>


### 2️⃣ Create Virtual Environment

python -m venv venv


Activate:

Mac/Linux:

source venv/bin/activate


Windows:

venv\Scripts\activate


### 3️⃣ Install Dependencies

pip install pandas numpy matplotlib seaborn scikit-learn joblib


### 4️⃣ Run Notebook
Open the `.ipynb` file in VS Code and run all cells sequentially.

---

## 📂 Project Structure

```text
project/
│
├── notebooks/
│   └── ames_housing.ipynb
│
├── models/
│   ├── ames_housing_model.joblib
│   ├── logistic_model.pkl
│   └── feature_columns.pkl
│
└── README.md
```


---

## 📌 Future Improvements

- Add Gradient Boosting / XGBoost comparison  
- Deploy using Streamlit  
- Add SHAP-based explainability  
- Integrate REST API using FastAPI  
- Automate feature selection  

---

## 🎯 Learning Outcomes

This project demonstrates:

- End-to-end ML pipeline development  
- Structured feature engineering  
- Proper evaluation methodology  
- Hyperparameter optimization  
- Deployment-ready model saving  
