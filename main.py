import pandas as pd
import kagglehub

kagglehub.login()

file_path = kagglehub.dataset_download("krzysztofjamroz/apartment-prices-in-poland", path="apartments_pl_2024_06.csv")

df = pd.read_csv(file_path)
print(df.head())