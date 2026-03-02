import pathlib

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "AmesHousing.csv"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Drop obvious identifier columns
    df = df.drop(columns=["Order", "PID"], errors="ignore")
    return df


@st.cache_resource(show_spinner=False)
def train_model(df: pd.DataFrame):
    # Using the real-world Ames Housing target column
    target_col = "SalePrice"

    # Simple, interpretable set of features inspired by the notebook
    feature_cols = [
        "Neighborhood",
        "Overall Qual",
        "Overall Cond",
        "Year Built",
        "Gr Liv Area",
        "Full Bath",
        "Bedroom AbvGr",
        "Kitchen Qual",
        "Exter Qual",
        "Central Air",
    ]

    df_model = df[feature_cols + [target_col]].dropna()

    X = df_model[feature_cols]
    y = df_model[target_col]

    numeric_features = [
        "Overall Qual",
        "Overall Cond",
        "Year Built",
        "Gr Liv Area",
        "Full Bath",
        "Bedroom AbvGr",
    ]
    categorical_features = [
        "Neighborhood",
        "Kitchen Qual",
        "Exter Qual",
        "Central Air",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    # Simple and explainable model for a student project
    model = LinearRegression()

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    valid_pred = pipeline.predict(X_valid)

    # Simple evaluation: MAE and R²
    mae = float((abs(valid_pred - y_valid)).mean())
    ss_res = float(((valid_pred - y_valid) ** 2).sum())
    ss_tot = float(((y_valid - y_valid.mean()) ** 2).sum())
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    return pipeline, mae, r2


def build_sidebar(df: pd.DataFrame):
    st.sidebar.header("Configure Scenario")
    st.sidebar.caption(
        "Tip: Use realistic combinations of quality, size and year "
        "so that price changes make intuitive sense."
    )

    neighborhoods = sorted(df["Neighborhood"].unique().tolist())
    neighborhood = st.sidebar.selectbox("Neighborhood", neighborhoods)

    overall_qual_min = int(df["Overall Qual"].min())
    overall_qual_max = int(df["Overall Qual"].max())
    overall_qual = st.sidebar.slider(
        "Overall Quality (1–10)",
        overall_qual_min,
        overall_qual_max,
        int(df["Overall Qual"].median()),
    )

    overall_cond_min = int(df["Overall Cond"].min())
    overall_cond_max = int(df["Overall Cond"].max())
    overall_cond = st.sidebar.slider(
        "Overall Condition (1–10)",
        overall_cond_min,
        overall_cond_max,
        int(df["Overall Cond"].median()),
    )

    min_size = int(df["Gr Liv Area"].quantile(0.1))
    max_size = int(df["Gr Liv Area"].quantile(0.9))
    default_size = int(df["Gr Liv Area"].median())
    size = st.sidebar.slider(
        "Above Ground Living Area (sq ft)",
        min_size,
        max_size,
        default_size,
        step=50,
    )

    min_year = int(df["Year Built"].min())
    max_year = int(df["Year Built"].max())
    year_built = st.sidebar.slider(
        "Year Built",
        min_year,
        max_year,
        int(df["Year Built"].median()),
    )

    full_bath_min = int(df["Full Bath"].min())
    full_bath_max = int(df["Full Bath"].max())
    full_bath = st.sidebar.slider(
        "Full Bathrooms",
        full_bath_min,
        full_bath_max,
        int(df["Full Bath"].median()),
    )

    bedroom_min = int(df["Bedroom AbvGr"].min())
    bedroom_max = int(df["Bedroom AbvGr"].max())
    bedrooms = st.sidebar.slider(
        "Bedrooms Above Ground",
        bedroom_min,
        bedroom_max,
        int(df["Bedroom AbvGr"].median()),
    )

    kitchen_qual_options = sorted(df["Kitchen Qual"].unique().tolist())
    kitchen_qual = st.sidebar.selectbox("Kitchen Quality", kitchen_qual_options)

    exter_qual_options = sorted(df["Exter Qual"].unique().tolist())
    exter_qual = st.sidebar.selectbox("Exterior Quality", exter_qual_options)

    central_air_options = sorted(df["Central Air"].unique().tolist())
    central_air = st.sidebar.selectbox("Central Air", central_air_options)

    return {
        "Neighborhood": neighborhood,
        "Overall Qual": overall_qual,
        "Overall Cond": overall_cond,
        "Year Built": year_built,
        "Gr Liv Area": size,
        "Full Bath": full_bath,
        "Bedroom AbvGr": bedrooms,
        "Kitchen Qual": kitchen_qual,
        "Exter Qual": exter_qual,
        "Central Air": central_air,
    }


def main():
    st.set_page_config(
        page_title="Intelligent Property Price Prediction",
        page_icon="🏡",
        layout="wide",
    )

    st.title("Intelligent Property Price Prediction")
    st.markdown(
        """
This app is a small project to predict house prices using a simple
machine learning model (Linear Regression) trained on the Ames Housing dataset.

Use the controls on the left to describe a property. The model will estimate
its sale price in US Dollars based on patterns it learned from the dataset.
"""
    )

    with st.spinner("Loading data and training model..."):
        df = load_data()
        model, mae, r2 = train_model(df)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Model Performance (Validation Set)")
        st.metric("Mean Absolute Error (MAE)", f"${mae:,.2f}")
        st.metric("R² Score", f"{r2:.3f}")

    with col2:
        st.subheader("Dataset Snapshot")
        st.write(f"Rows: **{df.shape[0]}**  |  Columns: **{df.shape[1]}**")
        st.dataframe(
            df[
                [
                    "Neighborhood",
                    "Overall Qual",
                    "Overall Cond",
                    "Year Built",
                    "Gr Liv Area",
                    "SalePrice",
                ]
            ].head(10)
        )

    st.markdown("---")
    st.subheader("Try Your Own Scenario")

    user_inputs = build_sidebar(df)

    if st.button("Predict Price", type="primary"):
        input_df = pd.DataFrame([user_inputs])
        predicted_price = float(model.predict(input_df)[0])

        st.markdown("### Estimated House Price")
        st.success(f"**${predicted_price:,.2f}**")

        st.caption(
            "Note: This is a data-driven estimate based on the Ames Housing dataset. "
            "It is not a substitute for professional property valuation."
        )


if __name__ == "__main__":
    main()

