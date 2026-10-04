import model_evaluation as me
from Data_precessing import load_invoice_data, apply_label, split_data, scale_data

import joblib
features = ['invoice_quantity','invoice_dollars','Freight','total_Quantity','total_item_dollars','avg_days_to_receive']
target = 'flag_invoice'


def main():
    df = load_invoice_data()
    df = apply_label(df)

    #prepare the data
    x_train, x_test, y_train, y_test = split_data(df, features, target)
    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

    # Train the model
    grid_search = me.train_random_forest_model(x_train_scaled, y_train)

    # Evaluate the model
    me.evaluate_model(grid_search, x_test_scaled, y_test)

    # Save the trained model
    joblib.dump(grid_search.best_estimator_, "../../models/random_forest_flagging_model.pkl")

if __name__ == "__main__":
    main()