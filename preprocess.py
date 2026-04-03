import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Copy of original DataFrame
    df_cleaned = df.copy()
    
    # Replace NaN with n/d
    categorical_cols_with_nans = ['condition', 'buildingMaterial', 'type']
    
    for col in categorical_cols_with_nans:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].fillna('n/d')
            
    return df_cleaned
