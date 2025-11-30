import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

# Set a consistent style for professional plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

def plot_regional_bar_chart(df: pd.DataFrame, metric_col: str, title: str):
    """
    Generates a bar chart showing a metric's average across different regions.

    Args:
        df (pd.DataFrame): DataFrame containing 'regional_indicator' and the metric.
        metric_col (str): The column name containing the metric to plot.
        title (str): The title of the chart.
    """
    if df.empty:
        print("Cannot plot: DataFrame is empty.")
        return

    plt.figure()
    # Use a custom color palette for a professional look
    sns.barplot(
        x=metric_col, 
        y='regional_indicator', 
        data=df, 
        palette=sns.color_palette("viridis", len(df))
    )
    plt.title(title, fontsize=16, pad=15)
    plt.xlabel(metric_col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel("Regional Indicator", fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_metric_time_series(df: pd.DataFrame, country_name: str, metric_col: str, title: str):
    """
    Generates a line plot showing a country's metric over time.

    Args:
        df (pd.DataFrame): Time series DataFrame (index is 'year', one column is the metric).
        country_name (str): Name of the country.
        metric_col (str): The column name of the metric to plot.
        title (str): The title of the chart.
    """
    if df.empty:
        print("Cannot plot: DataFrame is empty.")
        return

    plt.figure()
    plt.plot(df.index, df[metric_col], marker='o', linestyle='-', color='darkgreen')
    plt.title(title, fontsize=16, pad=15)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel(metric_col.replace('_', ' ').title(), fontsize=12)
    plt.xticks(df.index, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_scatter_correlation(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    """
    Generates a scatter plot to show the correlation between two metrics.
    
    Args:
        df (pd.DataFrame): The DataFrame with the data.
        x_col (str): Column name for the X-axis.
        y_col (str): Column name for the Y-axis.
        title (str): The title of the chart.
    """
    if df.empty:
        print("Cannot plot: DataFrame is empty.")
        return
        
    plt.figure()
    sns.regplot(x=x_col, y=y_col, data=df, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    plt.title(title, fontsize=16, pad=15)
    plt.xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    plt.ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    plt.tight_layout()
    plt.show()
