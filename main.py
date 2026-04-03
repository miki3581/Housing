from data_loader import load_data
from preprocess import clean_data, encode_features

def main():
    # Loading data
    df = load_data()
    
    # Cleaning data
    df_cleaned = clean_data(df)

    # Encoding categorical features
    df_encoded = encode_features(df_cleaned)

    print("Data shape after encoding:", df_encoded.shape)
    print("Sample columns:", df_encoded.columns[:10].tolist())
    print(df_encoded.head())
    
if __name__ == "__main__":
    main()