import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
DATA_FILE = 'apartments excel.xlsx'
OUTPUT_DIR = 'eda_plots'
TOP_N_CATEGORIES = 15 # For plots with high-cardinality categorical features

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created directory: {OUTPUT_DIR}")

# --- Load Data ---
try:
    print(f"Loading data from {DATA_FILE}...")
    df_eda = pd.read_excel(DATA_FILE, skiprows=1, header=0)
    print("Data loaded successfully.")
    print(f"Dataset shape: {df_eda.shape}")
    print("\nFirst 5 rows:")
    print(df_eda.head())
    print("\nColumn data types:")
    print(df_eda.info())
except FileNotFoundError:
    print(f"Error: Data file '{DATA_FILE}' not found. Make sure it's in the same directory as the script.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- Helper Function for Saving Plots ---
def save_plot(fig, filename_prefix):
    filepath = os.path.join(OUTPUT_DIR, f"{filename_prefix}.png")
    fig.savefig(filepath)
    print(f"Saved plot: {filepath}")
    plt.close(fig)

# --- Plotting Functions ---

# 1. Distribution of Target Variable (Price)
print("\n--- Plotting Target Variable Distribution ---")
fig_price, axs_price = plt.subplots(1, 2, figsize=(15, 6))
sns.histplot(df_eda['price'].dropna(), kde=True, ax=axs_price[0])
axs_price[0].set_title('Distribution of Price')
axs_price[0].set_xlabel('Price (EUR)')
axs_price[0].set_ylabel('Frequency')

sns.histplot(np.log1p(df_eda['price'].dropna()), kde=True, ax=axs_price[1]) # log1p handles 0s if any
axs_price[1].set_title('Distribution of Log-Transformed Price (log1p)')
axs_price[1].set_xlabel('Log(Price + 1)')
axs_price[1].set_ylabel('Frequency')
fig_price.tight_layout()
save_plot(fig_price, "01_price_distribution")

# 2. Histograms for Key Numerical Features
print("\n--- Plotting Numerical Feature Distributions ---")
numerical_features = ['surface', 'price_per_sqm', 'construction_year', 'rooms', 
                      'distance_school', 'distance_park', 'distance_transport', 
                      'distance_supermarket', 'latitude', 'longitude', 'accessibility_score']
for col in numerical_features:
    if col in df_eda.columns:
        fig_num, ax_num = plt.subplots(figsize=(8, 5))
        sns.histplot(df_eda[col].dropna(), kde=True, ax=ax_num)
        ax_num.set_title(f'Distribution of {col}')
        ax_num.set_xlabel(col)
        ax_num.set_ylabel('Frequency')
        save_plot(fig_num, f"02_hist_{col.replace(' ', '_')}")

# 3. Bar Plots for Key Categorical Features (Top N categories)
print("\n--- Plotting Categorical Feature Distributions ---")
categorical_features = ['heating_type', 'floor', 'property_status', 'property_type', 
                        'property form', 'sellerType', 'building_type', 'accessibility_label', 'rooms'] # rooms can also be seen as categorical here
for col in categorical_features:
    if col in df_eda.columns:
        fig_cat, ax_cat = plt.subplots(figsize=(10, 6))
        # For high cardinality, show top N and group others
        if df_eda[col].nunique() > TOP_N_CATEGORIES:
            top_categories = df_eda[col].value_counts().nlargest(TOP_N_CATEGORIES).index
            df_plot = df_eda.copy()
            df_plot[col] = df_plot[col].apply(lambda x: x if x in top_categories else 'Other')
            sns.countplot(y=col, data=df_plot, order=df_plot[col].value_counts().index, ax=ax_cat, palette="viridis")
            ax_cat.set_title(f'Distribution of {col} (Top {TOP_N_CATEGORIES} & Other)')
        else:
            sns.countplot(y=col, data=df_eda, order=df_eda[col].value_counts().index, ax=ax_cat, palette="viridis")
            ax_cat.set_title(f'Distribution of {col}')
        ax_cat.set_xlabel('Count')
        ax_cat.set_ylabel(col)
        fig_cat.tight_layout()
        save_plot(fig_cat, f"03_bar_{col.replace(' ', '_')}")

# 4. Scatter Plots: Numerical Features vs. Price
print("\n--- Plotting Numerical Features vs. Price ---")
scatter_vs_price_features = ['surface', 'price_per_sqm', 'construction_year', 'rooms', 'longitude', 'latitude', 'accessibility_score']
for col in scatter_vs_price_features:
    if col in df_eda.columns:
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=df_eda[col], y=df_eda['price'], ax=ax_scatter, alpha=0.5)
        ax_scatter.set_title(f'{col} vs. Price')
        ax_scatter.set_xlabel(col)
        ax_scatter.set_ylabel('Price (EUR)')
        save_plot(fig_scatter, f"04_scatter_{col.replace(' ', '_')}_vs_price")

# 5. Box Plots: Categorical Features vs. Price (Top N categories)
print("\n--- Plotting Categorical Features vs. Price ---")
boxplot_vs_price_features = ['heating_type', 'floor', 'property_status', 'property_type', 
                             'property form', 'sellerType', 'building_type', 'accessibility_label', 'rooms']
for col in boxplot_vs_price_features:
    if col in df_eda.columns:
        fig_box, ax_box = plt.subplots(figsize=(12, 7))
        # For high cardinality, show top N and group others
        if df_eda[col].nunique() > TOP_N_CATEGORIES:
            top_categories = df_eda[col].value_counts().nlargest(TOP_N_CATEGORIES).index
            df_plot = df_eda[df_eda[col].isin(top_categories)].copy() # Only plot top N for clarity
            sns.boxplot(x=col, y='price', data=df_plot, ax=ax_box, palette="Set2", order=top_categories)
            ax_box.set_title(f'Price Distribution by {col} (Top {TOP_N_CATEGORIES})')
        else:
            order = df_eda.groupby(col)['price'].median().sort_values().index # Order by median price
            sns.boxplot(x=col, y='price', data=df_eda, ax=ax_box, palette="Set2", order=order)
            ax_box.set_title(f'Price Distribution by {col}')
        ax_box.set_xlabel(col)
        ax_box.set_ylabel('Price (EUR)')
        plt.xticks(rotation=45, ha='right')
        fig_box.tight_layout()
        save_plot(fig_box, f"05_boxplot_{col.replace(' ', '_')}_vs_price")

# 6. Correlation Matrix Heatmap for Numerical Features
print("\n--- Plotting Correlation Matrix ---")
if df_eda.select_dtypes(include=np.number).shape[1] > 1: # Check if there are numerical columns
    numeric_df = df_eda.select_dtypes(include=np.number).dropna() # Drop rows with NaNs for correlation calculation
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr, vmin=-1, vmax=1)
        ax_corr.set_title('Correlation Matrix of Numerical Features')
        fig_corr.tight_layout()
        save_plot(fig_corr, "06_correlation_matrix")
    else:
        print("Skipping correlation matrix: No numeric data after dropping NaNs.")
else:
    print("Skipping correlation matrix: Not enough numerical features.")


# 7. Geospatial Scatter Plot (Simple)
print("\n--- Plotting Geospatial Distribution (Simple) ---")
if 'latitude' in df_eda.columns and 'longitude' in df_eda.columns:
    fig_geo, ax_geo = plt.subplots(figsize=(10, 8))
    # Use a subset for performance if dataset is very large, and drop NaNs
    df_sample_geo = df_eda[['latitude', 'longitude', 'price']].dropna().sample(n=min(1000, len(df_eda.dropna())), random_state=42)
    scatter_geo = ax_geo.scatter(df_sample_geo['longitude'], df_sample_geo['latitude'], 
                                 c=df_sample_geo['price'], cmap='viridis', alpha=0.6,
                                 s=np.log1p(df_sample_geo['price'])/5) # Size by log price for better viz
    ax_geo.set_title('Geospatial Distribution of Apartments (Colored by Price)')
    ax_geo.set_xlabel('Longitude')
    ax_geo.set_ylabel('Latitude')
    cbar = plt.colorbar(scatter_geo, ax=ax_geo, label='Price (EUR)')
    ax_geo.grid(True)
    save_plot(fig_geo, "07_geospatial_price_scatter")
else:
    print("Skipping geospatial plot: 'latitude' or 'longitude' not found.")

print("\n--- Exploratory Data Analysis Script Finished ---")
print(f"All plots saved in '{OUTPUT_DIR}' directory.") 