from data_loader import load_data
from preprocess import clean_data, engineer_features, encode_features_train, encode_features_test
from model import split_data, scale_data, train_linear_regression, evaluate_model, save_model, load_model
from visualize import plot_actual_vs_predicted, plot_residuals, plot_residuals_vs_predicted, shap_summary, feature_importance, shap_dependence 
from model_xgb import optimize_params, load_or_train_model
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    # Set to 'warszawa', 'szczecin', 'gdansk', etc. or None for the whole dataset
    TARGET_CITY = 'warszawa'  
    model_type = 'XGB'
    
    # Loading data
    #df = load_data()
    df = pd.read_csv("housing.csv")

    # Filtering by city if specified
    if TARGET_CITY:
        print(f"\nFiltering data exclusively for city: {TARGET_CITY.capitalize()}")
        df = df[df['city'] == TARGET_CITY].copy()
    else:
        print("\nUsing the entire dataset (all cities).")
        
    # Cleaning data
    df_cleaned = clean_data(df)

    # Feature engeenering
    df_engineered = engineer_features(df_cleaned)

    # Splitting X and y
    X_full = df_engineered.drop(columns = ['price'])
    y_full = df_engineered['price']

    # Splitting train and test
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size = 0.2, random_state = 42)

    # Reseting indexes
    X_train.reset_index(drop = True, inplace = True)
    y_train.reset_index(drop = True, inplace = True)
    X_test.reset_index(drop = True, inplace = True)
    y_test.reset_index(drop = True, inplace = True)

    # Encoding categorical features
    df_train_encoded, train_feature_col = encode_features_train(X_train)
    df_test_encoded = encode_features_test(X_test, train_feature_col)
    df_test_encoded = df_test_encoded.reindex(columns=df_train_encoded.columns, fill_value=0)

    if model_type == 'Lin_reg':

        X_train_lin = df_train_encoded
        X_test_lin = df_test_encoded
        
        # Scaling numerical features
        X_train_scaled, X_test_scaled, scaler = scale_data(X_train_lin, X_test_lin)

        # Training Linear Regression model
        model = train_linear_regression(X_train_scaled, y_train)

        # Saving the model and the scaler as a dictionary
        model_filename = "Lin_reg/housing_model.joblib"
        artifacts_to_save = {'model': model, 'scaler': scaler}
        save_model(artifacts_to_save, model_filename)
        
        # Loading the model and scaler (to demonstrate it works without retraining)
        loaded_artifacts = load_model(model_filename)
        loaded_model = loaded_artifacts['model']
        loaded_scaler = loaded_artifacts['scaler'] # This is ready to transform new data!

        # Evaluating model performance
        metrics = evaluate_model(loaded_model, X_test_scaled, y_test)
        print("\nModel Performance on Test Set:")
        for k,v in metrics.items():
            print(f' {k}: {v:.2f}')
        
        # Visualisation
        y_pred = loaded_model.predict(X_test_scaled)
        plot_actual_vs_predicted(y_test, y_pred, TARGET_CITY, model_type)
        
        plot_residuals(y_test, y_pred, TARGET_CITY, model_type)

        plot_residuals_vs_predicted(y_test, y_pred, TARGET_CITY, model_type)

    elif model_type == 'XGB':

        # Assigning X and y training sets
        X_train_XGB = df_train_encoded
        y_train_XGB = y_train.reset_index(drop = True)

        # Fitting model
        model, feature_names = load_or_train_model(X_train_XGB, y_train_XGB, retrain=False)

        # Evaluation
        metrics = evaluate_model(model, df_test_encoded, y_test)
        print("\nModel Performance on Test Set:")
        for k,v in metrics.items():
            print(f' {k}: {v:.2f}')

        # Visualsation
        y_pred = model.predict(df_test_encoded)
        feature_importance(model, TARGET_CITY)
        plot_actual_vs_predicted(y_test, y_pred, TARGET_CITY, model_type)
        plot_residuals(y_test, y_pred, TARGET_CITY, model_type)
        plot_residuals_vs_predicted(y_test, y_pred, TARGET_CITY, model_type)

        shap_summary(model, X_train_XGB, TARGET_CITY)
        shap_dependence(model, X_train_XGB, TARGET_CITY)


if __name__ == "__main__":
    main()