import pandas as pd

def filter_by_country(df: pd.DataFrame, country_list: list) -> pd.DataFrame:
    """
    Filters the DataFrame for a specific list of countries.
    
    Args:
        df (pd.DataFrame): The full DataFrame.
        country_list (list): A list of country names (e.g., ['Israel', 'United States']).
        
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    if df.empty:
        return pd.DataFrame()
        
    filtered_df = df[df['country_name'].isin(country_list)].copy()
    return filtered_df

def calculate_regional_averages(df: pd.DataFrame, metric: str = 'life_ladder') -> pd.DataFrame:
    """
    Calculates the average of a given metric per regional indicator across all years.

    Args:
        df (pd.DataFrame): The full DataFrame.
        metric (str): The column name to average (e.g., 'life_ladder', 'gdp_per_capita').

    Returns:
        pd.DataFrame: A DataFrame of regional averages, sorted high to low.
    """
    if df.empty:
        return pd.DataFrame()
        
    # Calculate the mean of the specified metric, grouping by region
    regional_avg = df.groupby('regional_indicator')[metric].mean().sort_values(ascending=False).reset_index()
    regional_avg = regional_avg.rename(columns={metric: f'avg_{metric}'})
    return regional_avg

def track_country_over_time(df: pd.DataFrame, country_name: str, metric: str = 'life_ladder') -> pd.DataFrame:
    """
    Extracts a country's metric performance over time.

    Args:
        df (pd.DataFrame): The full DataFrame.
        country_name (str): The name of the country.
        metric (str): The column name to track.

    Returns:
        pd.DataFrame: Time series data for the country and metric.
    """
    country_time_series = df[df['country_name'] == country_name][['year', metric]].set_index('year')
    return country_time_series
