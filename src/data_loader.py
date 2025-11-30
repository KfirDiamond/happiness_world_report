import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the World Happiness Report CSV file and performs initial cleaning.
    
    Args:
        file_path (str): The full path to the CSV file.

    Returns:
        pd.DataFrame: The loaded and cleaned DataFrame.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
        
        # Standardize column names for easier use
        df.columns = df.columns.str.replace(' ', '_').str.lower()
        
        # Rename long columns for brevity
        df = df.rename(columns={
            'log_gdp_per_capita': 'gdp_per_capita',
            'healthy_life_expectancy_at_birth': 'life_expectancy',
            'freedom_to_make_life_choices': 'freedom',
            'perceptions_of_corruption': 'corruption'
        })

        print("Data loaded and columns cleaned successfully.")
        return df
    
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return pd.DataFrame()

# Example of how this might be used (not run when imported)
if __name__ == '__main__':
    # Assuming the data is in a folder named 'data' relative to the project root
    # For testing, you would need to adjust the path
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    data_file = os.path.join(data_dir, 'World Happiness Report.csv')
    
    df = load_data(data_file)
    if not df.empty:
        print("\nInitial DataFrame structure:")
        print(df.head())
        print(df.info())
