import joblib
import pandas as pd
from pathlib import Path


# ============================================================
# MODEL PATH
# ============================================================

CURRENT_FILE = Path(__file__).resolve()

# inference -> backend -> project root
PROJECT_ROOT = CURRENT_FILE.parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_flagging_model.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the trained invoice flagging model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Invoice flagging model not found at:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# PREDICTION
# ============================================================

def predict_invoice_flag(input_data):
    """
    Predict whether an invoice should be flagged.

    Parameters
    ----------
    input_data : dict
        Dictionary containing the invoice features.

    Returns
    -------
    pandas.DataFrame
        Input data with predicted_invoice_flag column.
    """

    # Load trained model
    model = load_model()

    # Convert input to DataFrame
    input_df = pd.DataFrame(input_data)

    # Expected model features
    expected_features = [
        "invoice_quantity",
        "invoice_dollars",
        "Freight",
        "total_Quantity",
        "total_item_dollars",
        "avg_days_to_receive",
    ]

    # Check required columns
    missing_features = [
        feature
        for feature in expected_features
        if feature not in input_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Keep features in the exact training order
    model_input = input_df[expected_features]

    # Generate prediction
    prediction = model.predict(model_input)

    # Add prediction to result
    input_df["predicted_invoice_flag"] = prediction

    return input_df


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample_input = {
        "invoice_quantity": [10, 20],
        "invoice_dollars": [1000, 8000],
        "Freight": [50, 100],
        "total_Quantity": [15, 85],
        "total_item_dollars": [1200, 20],
        "avg_days_to_receive": [5, 15],
    }

    predictions = predict_invoice_flag(sample_input)

    print("\nInvoice Flag Predictions:")
    print(predictions)
