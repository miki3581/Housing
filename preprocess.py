import pandas as pd

# Cleaning data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Copy of original DataFrame
    df_cleaned = df.copy()
    
    # Replace null with n/d
    categorical_cols_with_nans = ['condition', 'buildingMaterial', 'type', 'hasElevator']
    
    for col in categorical_cols_with_nans:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].fillna('n/d')
            
    
    # buildYear - median depending on city and building type
    if 'buildYear' in df_cleaned.columns and 'city' in df_cleaned.columns and 'type' in df_cleaned.columns:
        df_cleaned['buildYear'] = df_cleaned.groupby(['city', 'type'])['buildYear'].transform(lambda x: x.fillna(x.median()))
        
        # If NaNs still present, fill with overall median
        df_cleaned['buildYear'] = df_cleaned['buildYear'].fillna(df_cleaned['buildYear'].median())

    # Distances - median depending on city
    distance_cols = [
        'schoolDistance', 'clinicDistance', 'postOfficeDistance', 
        'kindergartenDistance', 'restaurantDistance', 'collegeDistance', 
        'pharmacyDistance'
    ]
    for col in distance_cols:
        if col in df_cleaned.columns and 'city' in df_cleaned.columns:
            df_cleaned[col] = df_cleaned.groupby('city')[col].transform(lambda x: x.fillna(x.median()))
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

    # floorCount - mode depending on city and type
    if 'floorCount' in df_cleaned.columns and 'city' in df_cleaned.columns and 'type' in df_cleaned.columns:
        # lambda uses the first mode ([0]). If mode is empty (all NaNs in group), it returns float('nan')
        safe_mode = lambda x: x.fillna(x.mode()[0] if not x.mode().empty else float('nan'))
        df_cleaned['floorCount'] = df_cleaned.groupby(['city', 'type'])['floorCount'].transform(safe_mode)
        df_cleaned['floorCount'] = df_cleaned['floorCount'].fillna(df_cleaned['floorCount'].mode()[0])

    # floor - median depending on building type and floorCount
    if 'floor' in df_cleaned.columns and 'floorCount' in df_cleaned.columns and 'type' in df_cleaned.columns:
        # 1. Most precise: median for a specific building type and specific floor count
        df_cleaned['floor'] = df_cleaned.groupby(['type', 'floorCount'])['floor'].transform(lambda x: x.fillna(x.median()))
        
        # 2. Fallback: median for specific floor count regardless of building type
        df_cleaned['floor'] = df_cleaned.groupby('floorCount')['floor'].transform(lambda x: x.fillna(x.median()))
        
        # 3. Ultimate fallback: overall median
        df_cleaned['floor'] = df_cleaned['floor'].fillna(df_cleaned['floor'].median())

    return df_cleaned

# Encoding features
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df.copy()
    
    # Remove unnecessary columns
    if 'id' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=['id'])
        
    # Encode binary variables (yes/no to 1/0)
    binary_cols = ['hasParkingSpace', 'hasBalcony', 'hasSecurity', 'hasStorageRoom']
    binary_mapping = {'yes': 1, 'no': 0}
    for col in binary_cols:
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(binary_mapping)
            
    # One-Hot Encoding for remaining categorical variables
    categorical_cols = ['city', 'type', 'ownership', 'buildingMaterial', 'condition', 'hasElevator']
    cols_to_encode = [col for col in categorical_cols if col in df_encoded.columns]
    
    df_encoded = pd.get_dummies(df_encoded, columns=cols_to_encode, drop_first=True, dtype=int)
    
    return df_encoded
