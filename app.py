import gradio as gr
import pandas as pd
from catboost import CatBoostClassifier

# Load CatBoost model
model = CatBoostClassifier()
model.load_model("catboost_model.cbm", format="cbm")

# Prediction function
def predict_quality(pH, Hardness, Solids, Chloramines, Sulfate, Conductivity,
                    Organic_Carbon, Trihalomethanes, Turbidity):
    try:
        input_data = pd.DataFrame([[pH, Hardness, Solids, Chloramines, Sulfate,
                                     Conductivity, Organic_Carbon, Trihalomethanes, Turbidity]],
                                   columns=["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
                                            "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"])
        prediction = model.predict(input_data)[0]
        return "💧 Water is Safe to Drink" if prediction == 1 else "❌ Water is Not Safe to Drink"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Define Gradio UI
inputs = [
    gr.Number(label="pH"),
    gr.Number(label="Hardness"),
    gr.Number(label="Solids"),
    gr.Number(label="Chloramines"),
    gr.Number(label="Sulfate"),
    gr.Number(label="Conductivity"),
    gr.Number(label="Organic_Carbon"),  # Match column name in model
    gr.Number(label="Trihalomethanes"),
    gr.Number(label="Turbidity")
]

demo = gr.Interface(
    fn=predict_quality,
    inputs=inputs,
    outputs="text",
    title="💧 AquaMind: Water Quality Predictor",
    description="Enter water chemistry values to check if water is safe to drink."
)

demo.launch()
