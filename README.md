# 🏠 Ames Housing Price Prediction System  
## From Raw Real Estate Data to Production-Ready Machine Learning Pipeline  

---

## 📌 Project Overview

This project implements a complete **end-to-end Machine Learning regression system** to predict residential property prices using the Ames Housing dataset from OpenML.

You can explore the interactive web app here:  
👉 **Hosted Streamlit app:** `https://propertypriceprediction.streamlit.app/`

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

The notebook and app both evaluate the model on a **held‑out validation set**:

- **Train / validation split:** 80% training, 20% validation  
- **Metrics (shown in the Streamlit app):**
  - **Mean Absolute Error (MAE)** on the validation set  
  - **R² score** on the validation set  

The Streamlit UI surfaces these metrics at the top of the page so you can see how the model performs for the current training setup.  
For full, experiment‑level evaluation (cross‑validation, parameter sweeps, plots, etc.), see the Jupyter notebook in `notebooks/model_training.ipynb`.

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

## 💾 Model & Artifacts

Saved artifacts (for local experimentation and deployment) include:

| Path / File | Description |
|-------------|------------|
| `models/ames_housing_model.joblib` | Serialized model/pipeline trained on the Ames Housing dataset |
| `notebooks/model_training.ipynb` | End‑to‑end training, feature engineering, and evaluation workflow |

Model artifacts are created in the notebook using `joblib`/`scikit-learn` so you can reload and reuse the trained pipeline without retraining from scratch.

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

## 🚀 How to Run the Project Locally

### 1️⃣ Clone the repository

```bash
git clone <repository-link>
cd Property_Price_Prediction
```

### 2️⃣ Create and activate a virtual environment

**Mac / Linux**

```bash
python -m venv venv
source venv/bin/activate
```

**Windows (PowerShell / CMD)**

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

All required packages are listed in `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit app

From the project root:

```bash
streamlit run app.py
```

This will start a local server and open the app in your browser (or you can visit `http://localhost:8501` manually).

### 5️⃣ (Optional) Explore the training notebook

To inspect the full training and evaluation pipeline:

```bash
jupyter notebook notebooks/model_training.ipynb
```

Run the cells sequentially to reproduce the preprocessing, model training, and evaluation flow.

---

## 📂 Project Structure

```text
Property_Price_Prediction/
│
├── app.py                     # Streamlit web application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation (this file)
│
├── data/
│   └── AmesHousing.csv        # Raw Ames Housing dataset
│
├── notebooks/
│   └── model_training.ipynb   # End‑to‑end model training & evaluation
│
└── models/
    └── ames_housing_model.joblib   # Saved model/pipeline artifact
```


---

## 📌 Future Improvements

- Add Gradient Boosting / XGBoost comparison  
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
