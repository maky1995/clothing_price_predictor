from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from data_utils import FEATURES, load_dataset, prepare_data

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "Price_Quota_Model.pkl"
ENCODER_FILE = BASE_DIR / "price_encoder.pkl"


def train_and_save():
    df = prepare_data(load_dataset())
    X = df[FEATURES]
    y = df["initial_price"]

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_encoded = encoder.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_encoded, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(encoder, ENCODER_FILE)

    return model, encoder, df


@st.cache_resource
def load_model():
    if MODEL_FILE.exists() and ENCODER_FILE.exists():
        return joblib.load(MODEL_FILE), joblib.load(ENCODER_FILE), None
    return train_and_save()


@st.cache_data
def get_choices():
    df = prepare_data(load_dataset())
    return {
        col: sorted(df[col].dropna().astype(str).unique().tolist())
        for col in FEATURES
    }


@st.cache_data
def get_price_stats():
    df = prepare_data(load_dataset())
    return df["initial_price"].agg(["min", "max", "mean"]).to_dict()


st.set_page_config(
    page_title="Clothing Price Predictor",
    page_icon="👗",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stMetric { text-align: center; }
    div[data-testid="stMetricValue"] { font-size: 2.5rem; color: #B5395C; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("👗 Clothing Price Predictor")
st.write(
    "Pick the attributes of a clothing item below and a Random Forest model "
    "will estimate its initial listing price, trained on real Myntra catalog data."
)

try:
    model, encoder, _ = load_model()
    choices = get_choices()
    stats = get_price_stats()

    with st.sidebar:
        st.header("About")
        st.write(
            "This app uses a **Random Forest Regressor** trained on clothing "
            "attributes (category, gender, brand, fabric, pattern, occasion, "
            "and length) to predict an item's initial price in INR."
        )
        st.metric("Training price range", f"₹{stats['min']:,.0f} – ₹{stats['max']:,.0f}")
        st.caption(f"Average training price: ₹{stats['mean']:,.0f}")
        st.divider()
        st.caption("Model: RandomForestRegressor · Encoding: OneHotEncoder")

    st.subheader("Clothing Details")
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox("Category", choices["category"])
        brand = st.selectbox("Brand", choices["brand"])
        fabric = st.selectbox("Fabric", choices["fabric"])
        occasion = st.selectbox("Occasion", choices["occasion"])
    with col2:
        gender = st.selectbox("Gender", choices["gender"])
        pattern = st.selectbox("Pattern", choices["pattern"])
        length = st.selectbox("Length", choices["length"])

    predict_clicked = st.button(
        "💰 Predict Price", use_container_width=True, type="primary"
    )

    if predict_clicked:
        input_df = pd.DataFrame([{
            "category": category,
            "gender": gender,
            "brand": brand,
            "fabric": fabric,
            "pattern": pattern,
            "occasion": occasion,
            "length": length,
        }])

        encoded_input = encoder.transform(input_df)
        prediction = float(model.predict(encoded_input)[0])

        # Spread across the trees gives a rough sense of confidence.
        tree_preds = [t.predict(encoded_input)[0] for t in model.estimators_]
        low, high = min(tree_preds), max(tree_preds)

        st.markdown("---")
        st.subheader("Estimated Initial Price")
        st.metric("Predicted Price", f"₹{prediction:,.2f}")
        st.caption(f"Model's tree-level range: ₹{low:,.0f} – ₹{high:,.0f}")

except FileNotFoundError as e:
    st.error(str(e))
    st.info("Place Clothing_dataset.csv (or .xlsx) in this folder, then reload.")
except Exception as e:
    st.error(f"Something went wrong: {e}")
    st.info(
        "If you changed the dataset or scikit-learn version, delete the .pkl "
        "files and restart the app so they can be regenerated."
    )

st.markdown("---")
st.caption("Machine Learning Project • Random Forest Regression")
