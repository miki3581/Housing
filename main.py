from data_loader import load_data
from preprocess import clean_data, engineer_features, encode_features
from model import split_data, scale_data, train_linear_regression, evaluate_model, save_model, load_model
from visualize import plot_actual_vs_predicted, plot_residuals, plot_residuals_vs_predicted

def main():
    # --- Configuration ---
    # Set to 'warszawa', 'szczecin', 'gdansk', etc. or None for the whole dataset
    TARGET_CITY = 'warszawa'  
    
    # Loading data
    df = load_data()
    
    # Filtering by city if specified
    if TARGET_CITY:
        print(f"\nFiltering data exclusively for city: {TARGET_CITY.capitalize()}")
        df = df[df['city'] == TARGET_CITY].copy()
    else:
        print("\nUsing the entire dataset (all cities).")
        
    # Cleaning data
    df_cleaned = clean_data(df)

    # Feature Engineering
    df_engineered = engineer_features(df_cleaned)

    # Encoding categorical features
    df_encoded = encode_features(df_engineered)

    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = split_data(df_encoded, target_col='price')
    
    # Scaling numerical features
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # Training Linear Regression model
    model = train_linear_regression(X_train_scaled, y_train)

    # Saving the model and the scaler as a dictionary
    model_filename = "housing_model.joblib"
    artifacts_to_save = {'model': model, 'scaler': scaler}
    save_model(artifacts_to_save, model_filename)
    
    # Loading the model and scaler (to demonstrate it works without retraining)
    loaded_artifacts = load_model(model_filename)
    loaded_model = loaded_artifacts['model']
    loaded_scaler = loaded_artifacts['scaler'] # This is ready to transform new data!

    # Evaluating model performance
    metrics = evaluate_model(loaded_model, X_test_scaled, y_test)
    print("\nModel Performance on Test Set:")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.2f} PLN")
    print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f} PLN")
    print(f"R-squared (R2): {metrics['R2']:.4f}")
    
    # Visualisation
    y_pred = loaded_model.predict(X_test_scaled)
    plot_actual_vs_predicted(y_test, y_pred, TARGET_CITY)
    
    plot_residuals(y_test, y_pred, TARGET_CITY)

    plot_residuals_vs_predicted(y_test, y_pred, TARGET_CITY)

if __name__ == "__main__":
    main()