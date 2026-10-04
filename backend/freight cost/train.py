import joblib
import pathlib as path

from Data_preprocessing import load_vendor_data, prepare_features, split_data
from mode_evaluation import train_linear, train_decision_tree, train_random_forest, evaluate_model

def main():
    # Project root = Invoice Intelligence Machine Learning
    project_root = path.Path(__file__).resolve().parents[2]

    db_path = project_root / "Data" / "inventory.db"
    model_dir = project_root / "models"
    

    # Create models directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Database:", db_path)
    print("Models:", model_dir)

    # load vendor data
    df = load_vendor_data(str(db_path))

    # prepare features
    x, y = prepare_features(df)

    x_train, x_test, y_train, y_test = split_data(x, y)

    # train models
    linear_model = train_linear(x_train, y_train)
    decision_tree_model = train_decision_tree(x_train, y_train)
    random_forest_model = train_random_forest(x_train, y_train)

    # evaluate models
    results = []

    results.append(
        evaluate_model(
            linear_model, x_test, y_test, "Linear Regression"
        )
    )

    results.append(
        evaluate_model(
            decision_tree_model, x_test, y_test, "Decision Tree"
        )
    )

    results.append(
        evaluate_model(
            random_forest_model, x_test, y_test, "Random Forest"
        )
    )

    # select the best model based on R2 score
    best_model = max(results, key=lambda x: x["r2"])
    best_model_name = best_model["model_name"]

    best_mode = {
        "Linear Regression": linear_model,
        "Decision Tree": decision_tree_model,
        "Random Forest": random_forest_model
    }[best_model_name]


    # save the best model
    model_path = model_dir / f"{best_model_name}.pkl"

    joblib.dump(best_mode, model_path)

    print(f"\nBest model: {best_model_name}")
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    main()
