from data_loader import load_data
from preprocess import clean_data

def main():
    # Loading data
    df = load_data()
    
    # Cleaning data
    df_cleaned = clean_data(df)
    

if __name__ == "__main__":
    main()