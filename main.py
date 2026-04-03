from data_loader import load_data
from preprocess import clean_data, encode_features
from model import split_data, scale_data, train_linear_regression, evaluate_model

def main():
    # --- Configuration ---
    # Set to 'warszawa', 'szczecin', 'gdansk', etc. or None for the whole dataset
    TARGET_CITY = None 
    
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

    # Encoding categorical features
    df_encoded = encode_features(df_cleaned)

    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = split_data(df_encoded, target_col='price')
    
    # Scaling numerical features
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Training Linear Regression model
    model = train_linear_regression(X_train_scaled, y_train)
    
    # Evaluating model
    metrics = evaluate_model(model, X_test_scaled, y_test)
    
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:,.2f} PLN")
    print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:,.2f} PLN")
    print(f"R-squared (R2): {metrics['R2']:.4f}")

if __name__ == "__main__":
    main()