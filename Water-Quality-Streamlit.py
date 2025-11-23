import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_resource
def load_model():
    """
    Load the trained water quality model.
    Expects 'best_water_model.pkl' to be in the same folder as this file.
    """
    model_path = Path("best_water_model.pkl")
    if not model_path.exists():
        st.error(
            "Model file 'best_water_model.pkl' not found.\n\n"
            "Make sure it is in the same folder as this Streamlit app."
        )
        return None

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def main():
    st.set_page_config(
        page_title="AquaMind – Water Quality Checker",
        page_icon="💧",
        layout="centered",
    )

    st.title("💧 AquaMind – Water Quality Checker")
    st.write(
        """
        This app uses a machine learning model trained on real-world water quality data
        to predict whether water is **potable (safe to drink)** or **not potable**.
        """
    )

    model = load_model()
    if model is None:
        st.stop()

    st.subheader("Enter Water Quality Parameters")

    with st.form("water_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
            hardness = st.number_input("Hardness", min_value=0.0, value=150.0, step=1.0)
            solids = st.number_input("Solids", min_value=0.0, value=20000.0, step=100.0)

        with col2:
            chloramines = st.number_input(
                "Chloramines (ppm)", min_value=0.0, value=7.0, step=0.1
            )
            sulfate = st.number_input(
                "Sulfate (mg/L)", min_value=0.0, value=350.0, step=1.0
            )
            conductivity = st.number_input(
                "Conductivity (μS/cm)", min_value=0.0, value=500.0, step=1.0
            )

        with col3:
            organic_carbon = st.number_input(
                "Organic Carbon", min_value=0.0, value=10.0, step=0.1
            )
            trihalomethanes = st.number_input(
                "Trihalomethanes", min_value=0.0, value=60.0, step=0.1
            )
            turbidity = st.number_input(
                "Turbidity (NTU)", min_value=0.0, value=4.0, step=0.1
            )

        submitted = st.form_submit_button("Predict Water Potability")

    if submitted:
        # Arrange features in the correct order expected by the model
        feature_names = [
            "ph",
            "Hardness",
            "Solids",
            "Chloramines",
            "Sulfate",
            "Conductivity",
            "Organic_carbon",
            "Trihalomethanes",
            "Turbidity",
        ]

        input_data = pd.DataFrame(
            [
                [
                    ph,
                    hardness,
                    solids,
                    chloramines,
                    sulfate,
                    conductivity,
                    organic_carbon,
                    trihalomethanes,
                    turbidity,
                ]
            ],
            columns=feature_names,
        )

        try:
            # Try to get probability if available
            proba = None
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_data)[0][1]

            pred = model.predict(input_data)[0]

            # Handle different possible model outputs
            if isinstance(pred, (np.ndarray, list)):
                pred = pred[0]

            label = str(pred)
            potable = label in ["1", "Potable", "potable", "True", "true"]

            if potable:
                st.success("✅ Prediction: The water is **POTABLE (Safe to drink)**.")
            else:
                st.error("⚠️ Prediction: The water is **NOT POTABLE (Not safe to drink)**.")

            if proba is not None:
                st.write(f"Model confidence (potable): **{proba * 100:.2f}%**")
        except Exception as e:
            st.error(f"Error while making prediction: {e}")

    st.markdown("---")
    st.caption(
        "Built with Streamlit. Model file: `best_water_model.pkl`. "
        "Dataset features: pH, Hardness, Solids, Chloramines, Sulfate, "
        "Conductivity, Organic Carbon, Trihalomethanes, Turbidity."
    )


if __name__ == "__main__":
    main()
