import pandas as pd

parquet_file_path = '/PATH/TO/YOUR/data/Sweep/data/chunk-000/episode_000000.parquet'

try:
    # Read the parquet file into a pandas DataFrame
    df = pd.read_parquet(parquet_file_path)

    # Print the content of the DataFrame
    print(f"Successfully read parquet file: {parquet_file_path}")
    print("DataFrame head:")
    print(df.head())
    print("\nDataFrame info:")
    df.info()
    print("\nDataFrame description:")
    print(df.describe())

except FileNotFoundError:
    print(f"Error: The file was not found at {parquet_file_path}")
except Exception as e:
    print(f"An error occurred while reading the parquet file: {e}") 