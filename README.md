# ML Real Estate Price Prediction

A comprehensive machine learning project for predicting real estate prices in Romania using multiple algorithms and advanced feature engineering.

## Project Structure

```
ml-real-estate/
├── data/
│   ├── raw/                    # Original Excel datasets
│   ├── processed/             # Cleaned and preprocessed data
│   └── external/              # External data sources
├── src/
│   ├── data/                  # Data processing modules
│   ├── models/                # ML model implementations
│   ├── evaluation/            # Model evaluation utilities
│   ├── visualization/         # Plotting and visualization
│   └── utils/                 # Helper functions
├── notebooks/                 # Jupyter notebooks for exploration
├── experiments/               # Experiment tracking
├── results/                   # Model outputs and reports
├── configs/                   # Configuration files
├── scripts/                   # Standalone execution scripts
└── tests/                     # Unit tests
```

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Activate Virtual Environment** (if not already active)
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Run Initial Data Processing**
   ```bash
   python scripts/process_data.py
   ```

4. **Train All Models**
   ```bash
   python scripts/train_all_models.py
   ```

5. **Compare Results**
   ```bash
   python scripts/evaluate_models.py
   ```

## Data Sources

- **apartments excel.xlsx**: Main apartment listings dataset
- **Real estate sales from nov 2024-apr 2025.xlsx**: Recent sales data
- **localitati.csv**: Romanian localities for city mapping
- **market_trends_vanzari_ancpi.csv**: ANCPI market trends

## Machine Learning Models

The project tests multiple algorithms:

- **Linear Models**: Linear Regression, Ridge, Lasso, Elastic Net
- **Tree-Based**: Random Forest, XGBoost, LightGBM, CatBoost
- **Support Vector**: SVR
- **Ensemble**: Voting, Stacking
- **Neural Networks**: MLPRegressor (optional)

## Key Features

- 🏠 **Real Estate Specific**: Tailored for Romanian real estate market
- 📊 **Multiple Algorithms**: Compare 8+ ML algorithms
- 🎯 **Feature Engineering**: Advanced location and market features
- 📈 **Experiment Tracking**: Organized experiment management
- 🔍 **Model Interpretation**: SHAP values and feature importance
- 📋 **Automated Reports**: HTML reports with model comparisons

## Configuration

Edit configuration files in `configs/`:
- `data_config.yaml`: Data processing settings
- `model_configs.yaml`: Model hyperparameters
- `experiment_config.yaml`: Experiment settings

## Results

All results are stored in `results/`:
- `models/`: Trained model files
- `plots/`: Visualization outputs
- `reports/`: HTML analysis reports
- `predictions/`: Prediction outputs

## Contributing

1. Create feature branches for new models or features
2. Add unit tests for new functionality
3. Update documentation for new features
4. Follow the existing code structure

## License

This project is for educational and research purposes. 