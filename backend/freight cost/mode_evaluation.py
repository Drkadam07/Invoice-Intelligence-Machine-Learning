import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def train_linear(x_train,y_train):
    ''' train a linear regression model and return the model'''
    model = LinearRegression()
    model.fit(x_train,y_train)
    return model

def train_decision_tree(x_train,y_train):
    ''' train a decision tree model and return the model'''
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(x_train,y_train)
    return model

def train_random_forest(x_train,y_train):
    ''' train a random forest model and return the model'''
    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(x_train,y_train)
    return model


def evaluate_model(model, x_test, y_test, model_name: str):
    """Evaluate the model and return the metrics."""

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)

    # RMSE = square root of MSE
    rmse = mse ** 0.5

    mae = mean_absolute_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)

    print(f"\nModel: {model_name}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return {
        "model_name": model_name,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
