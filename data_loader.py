import pandas as pd
import kagglehub

#Loading dataset and returning it as a pandas DataFrame
def load_data() -> pd.DataFrame:

    file_path = kagglehub.dataset_download("krzysztofjamroz/apartment-prices-in-poland", path="apartments_pl_2024_06.csv")
    
    df = pd.read_csv(file_path)
    return df