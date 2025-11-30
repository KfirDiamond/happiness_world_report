# World Happiness Report Analysis

A comprehensive data analysis project examining global happiness levels and well-being indicators across countries from 2005 to 2022.

## Overview

This project analyzes the World Happiness Report data, published by the Sustainable Development Solutions Network and powered by Gallup World Poll data. The analysis explores how happiness and well-being vary across nations and identifies key factors that influence happiness scores at a global level.

The World Happiness Report emphasizes the importance of happiness and well-being as fundamental factors in government policy. It provides insights into global happiness patterns and explores how the science of happiness can inform policy decisions worldwide.

## Project Structure

```
happiness_world_report/
├── data/                          # Data directory
│   └── World Happiness Report.csv # Raw happiness data (2005-2022)
├── notebooks/                     # Jupyter notebooks for analysis
│   └── 8_Happiness.ipynb         # Main analysis notebook
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── data_loader.py             # Data loading and preprocessing
│   ├── analysis_pipeline.py       # Analysis workflows
│   └── visualization.py           # Plotting and visualization utilities
├── public/                        # Output and reports directory
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

## Features

- **Data Loading**: Efficient CSV parsing and data preprocessing
- **Analysis Pipeline**: Modular analysis workflows for happiness metrics
- **Visualizations**: Professional plots including:
  - Regional comparison bar charts
  - Time-series analysis
  - Correlation scatter plots
  - Distribution analysis
- **Jupyter Notebooks**: Interactive exploration and reporting

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd happiness_world_report
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Analysis

Launch the Jupyter notebook for interactive analysis:
```bash
jupyter notebook notebooks/8_Happiness.ipynb
```

### Using the Analysis Modules

```python
from src.data_loader import load_data
from src.analysis_pipeline import run_analysis
from src.visualization import plot_regional_bar_chart

# Load data
df = load_data('data/World Happiness Report.csv')

# Run analysis
results = run_analysis(df)

# Create visualizations
plot_regional_bar_chart(df, 'Life Ladder', 'Happiness Score by Region')
```

## Dependencies

- **pandas** (≥1.3.0): Data manipulation and analysis
- **matplotlib** (≥3.4.0): Plotting library
- **seaborn** (≥0.11.0): Statistical data visualization
- **jupyter** (≥1.0.0): Interactive notebooks
- **notebook** (≥6.4.0): Jupyter notebook interface

See `requirements.txt` for complete list.

## Data Source

The data comes from the World Happiness Report, which ranks countries by happiness levels based on survey responses from the Gallup World Poll. Key metrics include:
- Life Ladder (life satisfaction)
- Log GDP per capita
- Social support
- Healthy life expectancy
- Freedom to make life choices
- Generosity
- Perceptions of corruption

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is open source and available for educational and research purposes.

## Contact

For questions or inquiries, please open an issue in the repository.

---

**Last Updated**: November 2025  
**Data Coverage**: 2005-2022
