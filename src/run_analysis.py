"""Run the notebook-style analysis from the command line.

This script reproduces the main analysis flow in `notebooks/8_Happiness.ipynb`.
It delegates data loading and some analysis to `src` modules and saves plots to `public/`.

Usage:
    python -m src.run_analysis
    python src/run_analysis.py

Optional:
    Set environment variable SAVE_PLOTS=0 to disable saving (defaults to 1).
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from src.data_loader import load_data
from src.analysis_pipeline import calculate_regional_averages, track_country_over_time, filter_by_country
from src.visualization import plot_regional_bar_chart, plot_metric_time_series, plot_scatter_correlation


def ensure_public_dir():
    public_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'public')
    os.makedirs(public_dir, exist_ok=True)
    return public_dir


def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(project_root, 'data', 'World Happiness Report.csv')
    public_dir = ensure_public_dir()
    save_plots = os.environ.get('SAVE_PLOTS', '1') != '0'

    print(f"Loading data from: {data_file}")
    # load_data performs initial cleaning and standardizes column names
    happy_df = load_data(data_file)

    # Basic inspections
    print("\n=== HEAD ===")
    print(happy_df.head().to_string())
    print("\n=== DTYPES ===")
    print(happy_df.dtypes)
    print("\n=== NULL COUNTS ===")
    print(happy_df.isnull().sum())
    print("\n=== SHAPE ===")
    print(happy_df.shape)

    # Use cleaned column names (load_data replaces spaces with '_' and lowercases)
    # Typical cleaned columns: country_name, year, life_ladder, gdp_per_capita, social_support,
    # life_expectancy, freedom, generosity, corruption, positive_affect, negative_affect, regional_indicator

    # 1) Scatter/regression plots (Social support vs Life Ladder; GDP vs Life Ladder)
    if {'social_support', 'life_ladder'}.issubset(happy_df.columns):
        plt.figure(figsize=(8, 5))
        sns.regplot(data=happy_df, x='social_support', y='life_ladder', scatter_kws={'alpha': 0.5})
        plt.title('Social Support vs Life Ladder')
        if save_plots:
            out = os.path.join(public_dir, 'regplot_social_support_vs_life_ladder.png')
            plt.savefig(out, bbox_inches='tight')
            print(f"Saved: {out}")
        plt.close()

    if {'gdp_per_capita', 'life_ladder'}.issubset(happy_df.columns):
        plt.figure(figsize=(8, 5))
        sns.regplot(data=happy_df, x='gdp_per_capita', y='life_ladder', scatter_kws={'alpha': 0.5})
        plt.title('Log GDP Per Capita vs Life Ladder')
        if save_plots:
            out = os.path.join(public_dir, 'regplot_gdp_vs_life_ladder.png')
            plt.savefig(out, bbox_inches='tight')
            print(f"Saved: {out}")
        plt.close()

    # 2) Corruption pair and boxplot
    if {'gdp_per_capita', 'corruption', 'freedom'}.issubset(happy_df.columns):
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.regplot(data=happy_df, x='gdp_per_capita', y='corruption', scatter_kws={'alpha': 0.5}, ax=ax[0])
        sns.regplot(data=happy_df, x='freedom', y='corruption', scatter_kws={'alpha': 0.5}, ax=ax[1])
        if save_plots:
            out = os.path.join(public_dir, 'corruption_pair.png')
            fig.savefig(out, bbox_inches='tight')
            print(f"Saved: {out}")
        plt.close(fig)

    if 'corruption' in happy_df.columns:
        plt.figure(figsize=(4, 5))
        sns.boxplot(data=happy_df[['corruption']], color='#FFC0CB')
        if save_plots:
            out = os.path.join(public_dir, 'boxplot_corruption.png')
            plt.savefig(out, bbox_inches='tight')
            print(f"Saved: {out}")
        plt.close()

    # 3) Correlation heatmap for the main metrics (when available)
    candidate_cols = ['life_ladder', 'gdp_per_capita', 'social_support', 'life_expectancy', 'freedom',
                      'generosity', 'corruption', 'positive_affect', 'negative_affect', 'confidence_in_national_government']
    available = [c for c in candidate_cols if c in happy_df.columns]
    if available:
        corr_df = happy_df[available].corr().round(2)
        plt.figure(figsize=(10, 5))
        sns.heatmap(data=corr_df, cmap='coolwarm', annot=True)
        if save_plots:
            out = os.path.join(public_dir, 'heatmap_correlations.png')
            plt.savefig(out, bbox_inches='tight')
            print(f"Saved: {out}")
        plt.close()

    # 4) Top 10 overall and top 10 in 2022 (using cleaned names)
    if 'country_name' in happy_df.columns and 'life_ladder' in happy_df.columns:
        av_df = happy_df.groupby('country_name')[['life_ladder']].mean().sort_values('life_ladder', ascending=False).reset_index()
        df_2022 = happy_df[happy_df.year == 2022].sort_values('life_ladder', ascending=False).reset_index()
        fig, ax = plt.subplots(2, 1, figsize=(12, 7.5))
        plt.subplots_adjust(hspace=0.34)
        sns.barplot(data=av_df.head(10), x='life_ladder', y='country_name', ax=ax[0], palette='coolwarm')
        ax[0].set_title('Top Happiest Countries From 2005')
        ax[0].set_ylabel("")
        sns.barplot(data=df_2022.head(10), x='life_ladder', y='country_name', ax=ax[1], palette='coolwarm')
        ax[1].set_title('Top Happiest Countries In 2022')
        ax[1].set_ylabel("")
        if save_plots:
            out = os.path.join(public_dir, 'top10_happiest.png')
            fig.savefig(out, bbox_inches='tight')
            print(f"Saved: {out}")
        plt.close(fig)

    # 5) Country GDP time-series for selected countries
    if 'country_name' in happy_df.columns and 'gdp_per_capita' in happy_df.columns:
        countries = ['Israel', 'Canada', 'New Zealand', 'Switzerland', 'Denmark', 'Norway', 'Sweden', 'Netherlands', 'Iceland']
        fig, ax1 = plt.subplots(figsize=(10, 6.5))
        sns.set_style('darkgrid')
        colors = ['blue', 'black', 'red', 'green', 'gold', 'purple', 'pink', 'magenta', 'cyan']
        for c, col in zip(countries, colors):
            dfc = happy_df[happy_df['country_name'] == c]
            if not dfc.empty:
                ax1.plot(dfc['year'], dfc['gdp_per_capita'], label=c, color=col)
        ax1.set_xticks(np.arange(2005, 2023, 1))
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Log GDP Per Capita')
        ax1.set_title('Log GDP Per Capita (Yearly)')
        ax1.legend(loc='upper left')
        if save_plots:
            out = os.path.join(public_dir, 'gdp_timeseries_countries.png')
            fig.savefig(out, bbox_inches='tight')
            print(f"Saved: {out}")
        plt.close(fig)

    # 6) Neighborhood time series (life ladder and social support)
    if {'country_name', 'life_ladder', 'social_support'}.issubset(happy_df.columns):
        neigh_countries = ['Israel', 'Lebanon', 'Syria', 'Jordan', 'Egypt']
        fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))
        plt.subplots_adjust(hspace=0.3)
        sns.set_style('darkgrid')
        for c, col in zip(neigh_countries, ['blue', 'red', 'black', 'green', 'goldenrod']):
            d = happy_df[happy_df['country_name'] == c]
            if not d.empty:
                axes[0].plot(d['year'], d['life_ladder'], label=c, color=col)
                axes[1].plot(d['year'], d['social_support'], label=c, color=col)
        axes[0].set_xticks(np.arange(2005, 2023, 1))
        axes[0].set_title('Life Ladder')
        axes[0].legend()
        axes[1].set_xticks(np.arange(2005, 2023, 1))
        axes[1].set_title('Social Support')
        axes[1].legend()
        if save_plots:
            out = os.path.join(public_dir, 'neighborhood_timeseries.png')
            fig.savefig(out, bbox_inches='tight')
            print(f"Saved: {out}")
        plt.close(fig)

    # 7) Choropleth animation (Plotly) — using cleaned columns
    try:
        if {'country_name', 'life_ladder', 'year'}.issubset(happy_df.columns):
            fig = px.choropleth(happy_df.sort_values('year'),
                                locations='country_name',
                                color='life_ladder',
                                locationmode='country names',
                                template='plotly_dark',
                                color_continuous_scale='RdBu',
                                animation_frame='year')
            fig.update_layout(title='Life Ladder Comparison by Countries', height=600, width=800)
            if save_plots:
                out = os.path.join(public_dir, 'life_ladder_choropleth.html')
                fig.write_html(out)
                print(f"Saved: {out}")
    except Exception as e:
        print(f"Could not create choropleth: {e}")

    # 8) Regional Healthy Life Expectancy barplot
    if {'year', 'life_expectancy', 'regional_indicator', 'country_name'}.issubset(happy_df.columns):
        happy_df2 = happy_df.copy()
        happy_df2 = happy_df2[happy_df2.year > 2005]
        happy_df2 = happy_df2.replace(['South Asia', 'Central and Eastern Europe',
                                       'Middle East and North Africa', 'Latin America and Caribbean',
                                       'Commonwealth of Independent States', 'North America and ANZ',
                                       'Western Europe', 'Southeast Asia', 'East Asia'], 'Else')
        happy_df2.loc[happy_df2['country_name'] == 'Israel', 'regional_indicator'] = 'Israel'
        order = ['Sub-Saharan Africa', 'Else', 'Israel']
        happy_df2['regional_indicator'] = pd.Categorical(happy_df2['regional_indicator'], categories=order, ordered=True)
        color_palette = {'Israel': '#CCCCCC', 'Else': 'blue', 'Sub-Saharan Africa': 'red'}
        plt.figure(figsize=(9, 5.5))
        sns.set_style('darkgrid')
        ax = sns.barplot(data=happy_df2, x='year', y='life_expectancy', hue='regional_indicator', palette=color_palette)
        ax.set_ylim(45)
        ax.set_title('Healthy Life Expectancy At Birth Per Regional Indicator (Yearly)')
        if save_plots:
            out = os.path.join(public_dir, 'regional_life_expectancy_bar.png')
            plt.savefig(out, bbox_inches='tight')
            print(f"Saved: {out}")
        plt.close()

    print('\nAll done. Plots and outputs are saved in the `public/` directory.')


if __name__ == '__main__':
    main()
"""Run the notebook-style analysis from the command line.

This script reproduces the main analysis flow in `notebooks/8_Happiness.ipynb`.
It reads the raw CSV from `data/`, prints basic summaries, generates the same
plots and saves them to the `public/` directory (PNG or HTML for interactive).

Usage:
    python -m src.run_analysis
    python src/run_analysis.py

Optional:
    Set environment variable SAVE_PLOTS=0 to disable saving (defaults to 1).
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from src.data_loader import load_data
from src.analysis_pipeline import calculate_regional_averages, track_country_over_time, filter_by_country
from src.visualization import plot_regional_bar_chart, plot_metric_time_series, plot_scatter_correlation


def ensure_public_dir():
    public_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'public')
    os.makedirs(public_dir, exist_ok=True)
    return public_dir


def save_figure(fig, out_path: str):
    fig.savefig(out_path, bbox_inches='tight')


def main():
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_file = os.path.join(project_root, 'data', 'World Happiness Report.csv')
    public_dir = ensure_public_dir()
    save_plots = os.environ.get('SAVE_PLOTS', '1') != '0'

    print(f"Loading data from: {data_file}")
    # use the shared data loader (performs initial cleaning and column standardization)
    happy_df = load_data(data_file)

    # Notebook: initial inspections
    print("\n=== HEAD ===")
    print(happy_df.head().to_string())
    print("\n=== DTYPES ===")
    print(happy_df.dtypes)
    print("\n=== NULL COUNTS ===")
    print(happy_df.isnull().sum())
    print("\n=== SHAPE ===")
    print(happy_df.shape)

    # NOTE: `load_data` standardizes column names (spaces -> underscores, lowercased)
    # and renames some long columns. Use the standardized names below.
    # Mapping (example): 'social_support', 'life_ladder', 'gdp_per_capita', 'corruption', 'freedom', 'life_expectancy'

    # Regression plots: Social Support vs Life Ladder
    plt.figure(figsize=(8, 5))
    sns.regplot(data=happy_df, x='social_support', y='life_ladder', scatter_kws={'alpha': 0.5})
    plt.title('Social Support vs Life Ladder')
    if save_plots:
        out = os.path.join(public_dir, 'regplot_social_support_vs_life_ladder.png')
        plt.savefig(out, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close()

    # GDP vs Life Ladder (data_loader renamed log_gdp_per_capita -> gdp_per_capita)
    plt.figure(figsize=(8, 5))
    sns.regplot(data=happy_df, x='gdp_per_capita', y='life_ladder', scatter_kws={'alpha': 0.5})
    plt.title('Log GDP Per Capita vs Life Ladder')
    if save_plots:
        out = os.path.join(public_dir, 'regplot_gdp_vs_life_ladder.png')
        plt.savefig(out, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close()

    # Corruption pair: gdp_per_capita & freedom vs corruption
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.regplot(data=happy_df, x='gdp_per_capita', y='corruption', scatter_kws={'alpha': 0.5}, ax=ax[0])
    sns.regplot(data=happy_df, x='freedom', y='corruption', scatter_kws={'alpha': 0.5}, ax=ax[1])
    if save_plots:
        out = os.path.join(public_dir, 'corruption_pair.png')
        fig.savefig(out, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close(fig)

    # Boxplot corruption
    plt.figure(figsize=(4, 5))
    if 'corruption' in happy_df.columns:
        sns.boxplot(data=happy_df[['corruption']], color='#FFC0CB')
        if save_plots:
            out = os.path.join(public_dir, 'boxplot_corruption.png')
            plt.savefig(out, bbox_inches='tight')
            print(f"Saved: {out}")
    plt.close()
                                             
    # Heatmap of correlations (select columns using cleaned names where available)
    my_cols_clean = []
    col_map = {
        'Life Ladder': 'life_ladder',
        'Log GDP Per Capita': 'gdp_per_capita',
        'Social Support': 'social_support',
        'Healthy Life Expectancy At Birth': 'life_expectancy',
        'Freedom To Make Life Choices': 'freedom',
        'Generosity': 'generosity',
        'Perceptions Of Corruption': 'corruption',
        'Positive Affect': 'positive_affect',
        'Negative Affect': 'negative_affect',
        'Confidence In National Government': 'confidence_in_national_government'
    }
    for k, v in col_map.items():
        if v in happy_df.columns:
            my_cols_clean.append(v)
    if my_cols_clean:
        corr_df = happy_df[my_cols_clean].corr().round(2)
        plt.figure(figsize=(10, 5))
        sns.heatmap(data=corr_df, cmap='coolwarm', annot=True)
        if save_plots:
            out = os.path.join(public_dir, 'heatmap_correlations.png')
            plt.savefig(out, bbox_inches='tight')
            print(f"Saved: {out}")
        plt.close()

    # Top 10 average and 2022
    av_df = happy_df.groupby('Country Name')[['Life Ladder']].mean().sort_values('Life Ladder', ascending=False).reset_index()
    df_2022 = happy_df[happy_df.Year == 2022].sort_values('Life Ladder', ascending=False).reset_index()
    fig, ax = plt.subplots(2, 1, figsize=(12, 7.5))
    plt.subplots_adjust(hspace=0.34)
    sns.barplot(data=av_df.head(10), x='Life Ladder', y='Country Name', ax=ax[0], palette='coolwarm')
    ax[0].set_title('Top Happiest Countries From 2005')
    ax[0].set_ylabel("")
    sns.barplot(data=df_2022.head(10), x='Life Ladder', y='Country Name', ax=ax[1], palette='coolwarm')
    ax[1].set_title('Top Happiest Countries In 2022')
    ax[1].set_ylabel("")
    if save_plots:
        out = os.path.join(public_dir, 'top10_happiest.png')
        fig.savefig(out, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close(fig)

    # Country GDP time-series comparison (notebook selects specific countries)
    countries = ['Israel', 'Canada', 'New Zealand', 'Switzerland', 'Denmark', 'Norway', 'Sweden', 'Netherlands', 'Iceland']
    country_dfs = {c: happy_df[happy_df['Country Name'] == c] for c in countries}
    fig, ax1 = plt.subplots(figsize=(10, 6.5))
    sns.set_style('darkgrid')
    colors = ['blue', 'black', 'red', 'green', 'gold', 'purple', 'pink', 'magenta', 'cyan']
    for c, col in zip(countries, colors):
        dfc = country_dfs.get(c)
        if dfc is not None and not dfc.empty:
            ax1.plot(dfc['Year'], dfc['Log GDP Per Capita'], label=c, color=col)
    ax1.set_xticks(np.arange(2005, 2023, 1))
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Log GDP Per Capita')
    ax1.set_title('Log GDP Per Capita (Yearly)')
    ax1.legend(loc='upper left')
    if save_plots:
        out = os.path.join(public_dir, 'gdp_timeseries_countries.png')
        fig.savefig(out, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close(fig)

    # Neighborhood life ladder and social support
    neigh_countries = ['Israel', 'Lebanon', 'Syria', 'Jordan', 'Egypt']
    neigh = {c: happy_df[happy_df['Country Name'] == c] for c in neigh_countries}
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5))
    plt.subplots_adjust(hspace=0.3)
    sns.set_style('darkgrid')
    for c, col in zip(neigh_countries, ['blue', 'red', 'black', 'green', 'goldenrod']):
        d = neigh.get(c)
        if d is not None and not d.empty:
            axes[0].plot(d['Year'], d['Life Ladder'], label=c, color=col)
            axes[1].plot(d['Year'], d['Social Support'], label=c, color=col)
    axes[0].set_xticks(np.arange(2005, 2023, 1))
    axes[0].set_title('Life Ladder')
    axes[0].legend()
    axes[1].set_xticks(np.arange(2005, 2023, 1))
    axes[1].set_title('Social Support')
    axes[1].legend()
    if save_plots:
        out = os.path.join(public_dir, 'neighborhood_timeseries.png')
        fig.savefig(out, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close(fig)

    # Choropleth (Plotly) — save as HTML
    try:
        fig = px.choropleth(happy_df.sort_values('Year'),
                            locations='Country Name',
                            color='Life Ladder',
                            locationmode='country names',
                            template='plotly_dark',
                            color_continuous_scale='RdBu',
                            animation_frame='Year')
        fig.update_layout(title='Life Ladder Comparison by Countries', height=600, width=800)
        if save_plots:
            out = os.path.join(public_dir, 'life_ladder_choropleth.html')
            fig.write_html(out)
            print(f"Saved: {out}")
    except Exception as e:
        print(f"Could not create choropleth: {e}")

    # Regional Healthy Life Expectancy barplot
    happy_df2 = happy_df.copy()
    happy_df2 = happy_df2[happy_df2.Year > 2005]
    happy_df2 = happy_df2.replace(['South Asia', 'Central and Eastern Europe',
                                   'Middle East and North Africa', 'Latin America and Caribbean',
                                   'Commonwealth of Independent States', 'North America and ANZ',
                                   'Western Europe', 'Southeast Asia', 'East Asia'], 'Else')
    happy_df2.loc[happy_df2['Country Name'] == 'Israel', 'Regional Indicator'] = 'Israel'
    order = ['Sub-Saharan Africa', 'Else', 'Israel']
    happy_df2['Regional Indicator'] = pd.Categorical(happy_df2['Regional Indicator'], categories=order, ordered=True)
    color_palette = {'Israel': '#CCCCCC', 'Else': 'blue', 'Sub-Saharan Africa': 'red'}
    plt.figure(figsize=(9, 5.5))
    sns.set_style('darkgrid')
    ax = sns.barplot(data=happy_df2, x='Year', y='Healthy Life Expectancy At Birth', hue='Regional Indicator', palette=color_palette)
    ax.set_ylim(45)
    ax.set_title('Healthy Life Expectancy At Birth Per Regional Indicator (Yearly)')
    if save_plots:
        out = os.path.join(public_dir, 'regional_life_expectancy_bar.png')
        plt.savefig(out, bbox_inches='tight')
        print(f"Saved: {out}")
    plt.close()

    print('\nAll done. Plots and outputs are saved in the `public/` directory.')


if __name__ == '__main__':
    main()
