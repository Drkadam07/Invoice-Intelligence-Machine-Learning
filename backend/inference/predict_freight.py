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
    / "Linear Regression.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the trained freight cost model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Freight cost model not found at:\n{MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# PREDICTION
# ============================================================

def predict_freight_cost(input_data):
    """
    Predict freight cost.

    Parameters
    ----------
    input_data : dict
        Dictionary containing Quantity and Dollars.

    Returns
    -------
    pandas.DataFrame
        Input data with predicted_freight_cost column.
    """

    # Load trained model
    model = load_model()

    # Convert input to DataFrame
    input_df = pd.DataFrame(input_data)

    # Expected features
    expected_features = [
        "Quantity",
        "Dollars",
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

    # Keep features in exact training order
    model_input = input_df[expected_features]

    # Generate prediction
    prediction = model.predict(model_input)

    # Round prediction
    input_df["predicted_freight_cost"] = prediction.round(2)

    return input_df


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    sample_input = {
        "Quantity": [10, 20],
        "Dollars": [1000, 2000],
    }

    predictions = predict_freight_cost(sample_input)

    print("\nFreight Cost Predictions:")
    print(predictions)
