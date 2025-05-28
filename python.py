import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
RAW_DATA_FILE = 'apartments excel.xlsx' # Changed back to .xlsx
OUTPUT_CLEANED_DATA_FILE = 'cleaned_apartments_data.csv'
COLUMNS_TO_DROP_FINALLY = ['id', 'url', 'title', 'additional information', 'address'] # 'address' is dropped after city extraction

# --- Load Romanian Localities ---
def load_romanian_localities(csv_path='localitati.csv'):
    try:
        df_localitati = pd.read_csv(csv_path)
        if 'nume' in df_localitati.columns and 'judet' in df_localitati.columns:
            city_to_county_map = pd.Series(df_localitati.judet.values, index=df_localitati.nume).to_dict()
            localities_for_extraction = df_localitati['nume'].dropna().unique().tolist()
            localities_for_extraction = [loc for loc in localities_for_extraction if len(str(loc).strip()) > 3]
            localities_for_extraction.sort(key=lambda x: len(str(x)), reverse=True)
            print(f"Loaded {len(localities_for_extraction)} unique locality names for extraction and a map for {len(city_to_county_map)} cities to counties from {csv_path}.")
            return localities_for_extraction, city_to_county_map
        else:
            return get_fallback_cities(), {}
    except FileNotFoundError:
        return get_fallback_cities(), {}
    except Exception as e:
        return get_fallback_cities(), {}

def get_fallback_cities():
    cities = [
        "Bucuresti", "Cluj-Napoca", "Timisoara", "Iasi", "Constanta", "Craiova", "Brasov", "Galati",
        "Ploiesti", "Oradea", "Braila", "Arad", "Pitesti", "Sibiu", "Bacau", "Targu Mures",
        "Baia Mare", "Buzau", "Botosani", "Satu Mare", "Ramnicu Valcea", "Suceava", "Piatra Neamt",
        "Drobeta-Turnu Severin", "Focsani", "Targoviste", "Tulcea", "Resita", "Slatina",
        "Hunedoara", "Giurgiu", "Alexandria", "Zalau", "Sfantu Gheorghe", "Vaslui",
        "Calarasi", "Alba Iulia", "Slobozia", "Miercurea Ciuc", "Deva",
        "Mangalia", "Medgidia", "Turda", "Lugoj", "Roman", "Medias", "Pascani", "Onesti",
        "Sighisoara", "Petrosani", "Campina", "Campulung", "Fagaras", "Sebes", "Aiud",
        "Tecuci", "Carei", "Sighetu Marmatiei", "Radauti", "Gherla", "Dej"
    ]
    cities.sort(key=len, reverse=True)
    return cities

ROMANIAN_CITIES_FOR_EXTRACTION, CITY_TO_COUNTY_MAP = load_romanian_localities()
if not ROMANIAN_CITIES_FOR_EXTRACTION:
    ROMANIAN_CITIES_FOR_EXTRACTION = get_fallback_cities()

def standardize_judet_name(name):
    if not isinstance(name, str):
        return 'unknown'
    name_lower = name.lower()
    replacements = {'ă': 'a', 'â': 'a', 'î': 'i', 'ș': 's', 'ț': 't', 'ş': 's', 'ţ': 't'}
    for char, replacement in replacements.items():
        name_lower = name_lower.replace(char, replacement)
    name_lower = name_lower.replace('-', ' ')
    if name_lower.startswith("judetul "):
        name_lower = name_lower[len("judetul "):]
    return name_lower.strip()

def extract_city(address_str, city_list):
    if not isinstance(address_str, str):
        return 'Unknown'
    address_lower = address_str.lower()
    for city in city_list:
        if city.lower() in address_lower:
            return city
    return 'Unknown'

try:
    print(f"Loading data from {RAW_DATA_FILE}...")
    df = pd.read_excel(RAW_DATA_FILE, skiprows=1, header=0)
except FileNotFoundError:
    print(f"Error: Data file '{RAW_DATA_FILE}' not found. Make sure it's in the same directory as the script.")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

def replace_nan_strings(value):
    if isinstance(value, str) and value.lower() in ['nan', 'na', '-', '', ' ', 'unknown', 'fără informații']:
        return np.nan
    return value

for col in df.columns:
    df[col] = df[col].apply(replace_nan_strings)

if 'floor' in df.columns:
    floor_mapping = {'parter înalt': 0.5, 'parter': 0, 'demisol': -1, 'subsol': -2, 'mansardă': 99, 'ultimul etaj': 98}
    df['floor'] = df['floor'].replace(floor_mapping)
    df['floor'] = pd.to_numeric(df['floor'], errors='coerce')
if 'surface' in df.columns:
    df['surface'] = pd.to_numeric(df['surface'], errors='coerce')
if 'price_per_sqm' in df.columns:
    df['price_per_sqm'] = pd.to_numeric(df['price_per_sqm'], errors='coerce')
if 'rooms' in df.columns:
    df['rooms'] = pd.to_numeric(df['rooms'], errors='coerce')
if 'construction_year' in df.columns:
    df['construction_year'] = pd.to_numeric(df['construction_year'], errors='coerce')

for col_to_impute_early in ['surface', 'rooms', 'construction_year']:
    if col_to_impute_early in df.columns and df[col_to_impute_early].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col_to_impute_early]):
            median_val = df[col_to_impute_early].median()
            if pd.notna(median_val):
                df[col_to_impute_early] = df[col_to_impute_early].fillna(median_val)
            else:
                df[col_to_impute_early] = df[col_to_impute_early].fillna(0)

if 'surface' in df.columns and 'rooms' in df.columns:
    df['avg_room_size'] = df['surface'] / df['rooms'].replace(0, np.nan)
    df['avg_room_size'] = df['avg_room_size'].replace([np.inf, -np.inf], np.nan)
if 'construction_year' in df.columns:
    bins = [0, 1960, 1980, 2000, 2010, 2020, np.inf]
    labels = ['pre-1960', '1960-1979', '1980-1999', '2000-2009', '2010-2019', 'post-2019']
    df['year_built_category'] = pd.cut(df['construction_year'], bins=bins, labels=labels, right=False)

if 'address' in df.columns:
    df['city'] = df['address'].apply(lambda x: extract_city(x, ROMANIAN_CITIES_FOR_EXTRACTION))
    if CITY_TO_COUNTY_MAP:
        df['Județ'] = df['city'].map(CITY_TO_COUNTY_MAP)
        df['Județ'] = df['Județ'].fillna('Unknown')
        df['Județ'] = df['Județ'].apply(standardize_judet_name)
        print(f"Unique Județ values in main df after mapping and standardization: {df['Județ'].unique().tolist()}")
    else:
        df['Județ'] = 'unknown'
else:
    if 'city' not in df.columns:
        df['city'] = 'Unknown'
    if 'Județ' not in df.columns:
        df['Județ'] = 'unknown'

ANCPI_TREND_FILE = 'market_trends_vanzari_ancpi.csv'
try:
    df_ancpi_trends = pd.read_csv(ANCPI_TREND_FILE)
    df_ancpi_trends['Județ'] = df_ancpi_trends['Județ'].apply(standardize_judet_name)
    print(f"\nLoaded ANCPI trend data from {ANCPI_TREND_FILE}. Shape: {df_ancpi_trends.shape}")
    print(f"Unique Județ values in ANCPI trends file after standardization: {df_ancpi_trends['Județ'].unique().tolist()}")
    df_april_sales = df_ancpi_trends[(df_ancpi_trends['An'] == 2025) & (df_ancpi_trends['Luna'] == 4)].copy()
    if not df_april_sales.empty:
        df_april_sales = df_april_sales[['Județ', 'Valoare_vanzari']]
        df_april_sales.rename(columns={'Valoare_vanzari': 'county_sales_april_2025'}, inplace=True)
        original_rows = len(df)
        df = pd.merge(df, df_april_sales, on='Județ', how='left')
        print(f"Merged April 2025 county sales data. DF shape before merge: ({original_rows}, X), after: {df.shape}")
        print(f"NaNs in 'county_sales_april_2025' after merge: {df['county_sales_april_2025'].isnull().sum()}")
        df['county_sales_april_2025_log'] = np.log1p(df['county_sales_april_2025'])
        df.drop(columns=['county_sales_april_2025'], inplace=True)
    else:
        df['county_sales_april_2025_log'] = np.nan
except FileNotFoundError:
    df['county_sales_april_2025_log'] = np.nan
except Exception as e:
    df['county_sales_april_2025_log'] = np.nan

existing_cols_to_drop_now = [col for col in COLUMNS_TO_DROP_FINALLY if col in df.columns]
if existing_cols_to_drop_now:
    df.drop(columns=existing_cols_to_drop_now, inplace=True)

if 'price' in df.columns:
    TARGET_COL = 'price'
    numerical_cols_for_processing = df.select_dtypes(include=np.number).columns.tolist()
    if TARGET_COL in numerical_cols_for_processing:
        numerical_cols_for_processing.remove(TARGET_COL)
    cols_to_exclude_from_capping = ['construction_year'] + [col for col in df.columns if 'has_' in col and df[col].nunique(dropna=False) <= 3]
    numerical_cols_for_capping = [col for col in numerical_cols_for_processing if col not in cols_to_exclude_from_capping and df[col].nunique(dropna=False) > 2]
    categorical_cols_for_processing = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'city' in df.columns and 'city' not in categorical_cols_for_processing:
        categorical_cols_for_processing.append('city')
    if 'year_built_category' in df.columns and 'year_built_category' not in categorical_cols_for_processing:
         if pd.api.types.is_categorical_dtype(df['year_built_category']) or pd.api.types.is_object_dtype(df['year_built_category']):
            categorical_cols_for_processing.append('year_built_category')
else:
    print("Target column 'price' not found. Cannot proceed with structured cleaning.")
    exit()

for col in numerical_cols_for_capping:
    if col in df.columns and df[col].isnull().sum() < len(df[col]):
        lower_bound = df[col].quantile(0.01)
        upper_bound = df[col].quantile(0.99)
        if col == 'rooms':
            lower_bound = max(lower_bound, 1.0)
            if upper_bound <= lower_bound:
                upper_bound = lower_bound + 1
        if col == 'avg_room_size':
            lower_bound = max(lower_bound, 5.0)
            if upper_bound <= lower_bound:
                 if pd.notna(upper_bound) and pd.notna(lower_bound):
                    upper_bound = max(upper_bound, lower_bound + df[col].std() if pd.notna(df[col].std()) else lower_bound + 10)
                 elif pd.notna(lower_bound):
                    upper_bound = lower_bound + (df[col].std() if pd.notna(df[col].std()) and df[col].std() > 0 else 10)
        if pd.notna(lower_bound) and pd.notna(upper_bound) and lower_bound < upper_bound:
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

df_cleaned_for_saving = df.copy()
for col in numerical_cols_for_processing:
    if col in df_cleaned_for_saving.columns and df_cleaned_for_saving[col].isnull().any():
        median_val = df_cleaned_for_saving[col].median()
        if pd.notna(median_val):
            df_cleaned_for_saving[col] = df_cleaned_for_saving[col].fillna(median_val)
        else:
            df_cleaned_for_saving[col] = df_cleaned_for_saving[col].fillna(0)
for col in categorical_cols_for_processing:
    if col in df_cleaned_for_saving.columns and df_cleaned_for_saving[col].isnull().any():
        mode_val_series = df_cleaned_for_saving[col].mode()
        if not mode_val_series.empty:
            mode_val = mode_val_series[0]
            df_cleaned_for_saving[col] = df_cleaned_for_saving[col].fillna(mode_val)
        else:
            df_cleaned_for_saving[col] = df_cleaned_for_saving[col].fillna('Unknown_Mode')

try:
    df_cleaned_for_saving.to_csv(OUTPUT_CLEANED_DATA_FILE, index=False)
except Exception as e:
    print(f"Error saving cleaned data: {e}")

df_for_modeling = df_cleaned_for_saving.copy()
if TARGET_COL not in df_for_modeling.columns:
    print(f"CRITICAL ERROR: Target column '{TARGET_COL}' is missing from df_for_modeling. Columns available: {df_for_modeling.columns.tolist()}")
    exit()

df_for_modeling[TARGET_COL + '_log'] = np.log1p(df_for_modeling[TARGET_COL])
y = df_for_modeling[TARGET_COL + '_log']
X = df_for_modeling.drop(columns=[TARGET_COL, TARGET_COL + '_log'])
original_X_columns = X.columns.tolist()

categorical_cols_model = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols_model = X.select_dtypes(include=np.number).columns.tolist()

for col in numerical_cols_model:
    if X[col].isnull().any():
        median_val = X[col].median()
        if pd.notna(median_val):
            X[col] = X[col].fillna(median_val)
        else:
            X[col] = X[col].fillna(0)

encoders = {}
for col in categorical_cols_model:
    if col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str).values.ravel())
        encoders[col] = le

numerical_medians_model = {col: X[col].median() for col in numerical_cols_model if col in X.columns and pd.notna(X[col].median())}
for col in numerical_cols_model:
    if col not in numerical_medians_model:
        numerical_medians_model[col] = 0
categorical_modes_model = {}
for col_original_name in categorical_cols_model:
    if col_original_name in df_for_modeling.columns:
        mode_val_series = df_for_modeling[col_original_name].mode()
        if not mode_val_series.empty:
            categorical_modes_model[col_original_name] = mode_val_series[0]
        else:
            categorical_modes_model[col_original_name] = 'Unknown_Mode'

april_sales_map = {}
if 'df_april_sales' in locals() and isinstance(df_april_sales, pd.DataFrame) and not df_april_sales.empty:
    try:
        april_sales_map = pd.Series(df_april_sales.county_sales_april_2025.values, index=df_april_sales.Județ).to_dict()
    except AttributeError:
        april_sales_map = {}
    except Exception as e:
        april_sales_map = {}
if 'county_sales_april_2025_log' in X.columns:
    median_val_cs = X['county_sales_april_2025_log'].median()
    if pd.notna(median_val_cs):
        numerical_medians_model['county_sales_april_2025_log'] = median_val_cs
    else:
        numerical_medians_model['county_sales_april_2025_log'] = 0
elif 'county_sales_april_2025_log' in original_X_columns:
    numerical_medians_model['county_sales_april_2025_log'] = 0
else:
    numerical_medians_model['county_sales_april_2025_log'] = 0
if 'county_sales_april_2025_log' in X.columns and 'county_sales_april_2025_log' not in original_X_columns:
    original_X_columns.append('county_sales_april_2025_log')
    if 'county_sales_april_2025' in original_X_columns:
        original_X_columns.remove('county_sales_april_2025')

X_train, X_test, y_train_log, y_test_log = train_test_split(X, y, test_size=0.2, random_state=42)
y_test_original = np.expm1(y_test_log)

xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42, early_stopping_rounds=50)
try:
    eval_set = [(X_test, y_test_log)]
    xgb_reg.fit(X_train, y_train_log, eval_set=eval_set, verbose=False)
except Exception as e:
    print(f"Error during XGBoost model training: {e}")
    exit()

try:
    y_pred_log = xgb_reg.predict(X_test)
    y_pred_original = np.expm1(y_pred_log)
except Exception as e:
    print(f"Error during XGBoost prediction: {e}")
    exit()

mae = mean_absolute_error(y_test_original, y_pred_original)
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_original, y_pred_original)

print("\nModel Evaluation (on original price scale):")
print(f"  Mean Absolute Error (MAE): {mae:.2f} EUR")
print(f"  Root Mean Squared Error (RMSE): {rmse:.2f} EUR")
print(f"  R-squared (R2 ):          {r2:.4f}")

residuals_original_scale = y_test_original - y_pred_original
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_original, y=y_pred_original, alpha=0.5)
plt.plot([min(y_test_original.min(), y_pred_original.min()), max(y_test_original.max(), y_pred_original.max())], [min(y_test_original.min(), y_pred_original.min()), max(y_test_original.max(), y_pred_original.max())], 'k--', lw=2)
plt.xlabel("Actual Prices (EUR)")
plt.ylabel("Predicted Prices (EUR)")
plt.title("Actual vs. Predicted Prices (Original Scale)")
plt.savefig("actual_vs_predicted.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_original, y=residuals_original_scale, alpha=0.5)
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.xlabel("Predicted Prices (EUR)")
plt.ylabel("Residuals (Actual - Predicted EUR)")
plt.title("Residuals vs. Predicted Prices (Original Scale)")
plt.savefig("residuals_vs_predicted.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(residuals_original_scale, kde=True)
plt.xlabel("Residuals (EUR)")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals (Original Scale)")
plt.savefig("residuals_histogram.png")
plt.close()

print("\n\n--- Advanced Model Evaluation ---")
print("\n1. Residual Quantiles:")
abs_residuals = np.abs(residuals_original_scale)
percentile_95_abs_error = np.percentile(abs_residuals, 95)
print(f"  95th percentile of Absolute Errors: {percentile_95_abs_error:.2f} EUR")
median_abs_error = np.median(abs_residuals)
print(f"  Median Absolute Error: {median_abs_error:.2f} EUR")

X_test_original_features = df_for_modeling.loc[X_test.index].copy()
test_performance_df = pd.DataFrame({'actual_price': y_test_original, 'predicted_price': y_pred_original, 'residual': residuals_original_scale, 'abs_residual': abs_residuals})
test_performance_df = test_performance_df.join(X_test_original_features)

# print("\n2. Error by Segment:")
# print("\n2a. By Price Bracket (Actual Price):")
# price_bins = [0, 75000, 150000, 300000, np.inf]
# price_labels = ['< €75k', '€75k-€150k', '€150k-€300k', '> €300k']
# test_performance_df['price_bracket'] = pd.cut(test_performance_df['actual_price'], bins=price_bins, labels=price_labels, right=False)
# price_segment_metrics = test_performance_df.groupby('price_bracket', observed=False).agg(sample_count=('actual_price', 'count'), MAE=('abs_residual', 'mean'), RMSE=('residual', lambda x: np.sqrt(np.mean(x**2)))).reset_index()
# print(price_segment_metrics)
# if 'property_type' in X_test_original_features.columns:
#     print("\n2b. By Property Type:")
#     test_performance_df['property_type_analysis'] = test_performance_df['property_type'].fillna('Unknown')
#     property_type_segment_metrics = test_performance_df.groupby('property_type_analysis', observed=False).agg(sample_count=('actual_price', 'count'), MAE=('abs_residual', 'mean'), RMSE=('residual', lambda x: np.sqrt(np.mean(x**2)))).reset_index()
#     print(property_type_segment_metrics)

# print("\n3. Investigate Outliers (Top 5% Largest Absolute Residuals):")
# outlier_threshold_value = test_performance_df['abs_residual'].quantile(0.95)
# top_outliers_df = test_performance_df[test_performance_df['abs_residual'] >= outlier_threshold_value].sort_values(by='abs_residual', ascending=False)
# print(f"  Displaying properties with absolute residual >= {outlier_threshold_value:.2f} EUR (Top ~5%):")
# cols_to_show_for_outliers = ['actual_price', 'predicted_price', 'residual', 'surface', 'rooms', 'city', 'construction_year', 'property_type', 'price_per_sqm']
# existing_cols_to_show = [col for col in cols_to_show_for_outliers if col in top_outliers_df.columns]
# if not top_outliers_df.empty:
#     print(top_outliers_df[existing_cols_to_show].head(10))

def predict_new_apartment(apartment_data, model, 
                          num_cols_model_list, cat_cols_model_list, 
                          num_medians_dict, cat_modes_dict, encoders_dict, 
                          original_cols_order_list, 
                          city_list_for_extraction,
                          city_to_county_map,
                          april_sales_map_lookup, # Existing map for current/April 2025 sales
                          future_raw_county_sales=None): # New parameter for future sales value
    new_df = pd.DataFrame([apartment_data])
    for col in new_df.columns:
        new_df[col] = new_df[col].apply(replace_nan_strings)
    if 'floor' in new_df.columns:
        floor_mapping = {'parter înalt': 0.5, 'parter': 0, 'demisol': -1, 'subsol': -2, 'mansardă': 99, 'ultimul etaj': 98}
        new_df['floor'] = new_df['floor'].replace(floor_mapping)
        new_df['floor'] = pd.to_numeric(new_df['floor'], errors='coerce')
    if 'surface' in new_df.columns:
        new_df['surface'] = pd.to_numeric(new_df['surface'], errors='coerce')
    if 'rooms' in new_df.columns:
        new_df['rooms'] = pd.to_numeric(new_df['rooms'], errors='coerce')
    if 'construction_year' in new_df.columns:
        new_df['construction_year'] = pd.to_numeric(new_df['construction_year'], errors='coerce')
    if 'surface' in new_df.columns and 'rooms' in new_df.columns and 'avg_room_size' in original_cols_order_list:
        rooms_val = new_df['rooms'].iloc[0]
        if pd.isna(rooms_val): rooms_val = num_medians_dict.get('rooms', 1)
        rooms_val = max(float(rooms_val), 1.0)
        surface_val = new_df['surface'].iloc[0]
        if pd.isna(surface_val): surface_val = num_medians_dict.get('surface', 0)
        surface_val = float(surface_val)
        if pd.notna(surface_val) and pd.notna(rooms_val) and rooms_val > 0:
            new_df['avg_room_size'] = surface_val / rooms_val
        else:
            new_df['avg_room_size'] = num_medians_dict.get('avg_room_size', 0)
    elif 'avg_room_size' in original_cols_order_list:
         new_df['avg_room_size'] = num_medians_dict.get('avg_room_size', 0)
    if 'construction_year' in new_df.columns and 'year_built_category' in original_cols_order_list:
        year_val = new_df['construction_year'].iloc[0]
        if pd.isna(year_val): year_val = num_medians_dict.get('construction_year', 2000)
        bins = [0, 1960, 1980, 2000, 2010, 2020, np.inf]
        labels = ['pre-1960', '1960-1979', '1980-1999', '2000-2009', '2010-2019', 'post-2019']
        new_df['year_built_category'] = pd.cut(pd.Series([float(year_val)]), bins=bins, labels=labels, right=False)[0]
    elif 'year_built_category' in original_cols_order_list:
        new_df['year_built_category'] = cat_modes_dict.get('year_built_category', 'Unknown_Mode')
    if 'address' in new_df.columns and pd.notna(new_df['address'].iloc[0]):
        extracted_city = extract_city(new_df['address'].iloc[0], city_list_for_extraction)
        new_df['city'] = extracted_city
    elif 'city' not in new_df.columns or pd.isna(new_df['city'].iloc[0]):
        new_df['city'] = 'Unknown'
    current_city = new_df['city'].iloc[0]
    if city_to_county_map and current_city != 'Unknown':
        mapped_judet = city_to_county_map.get(current_city, 'Unknown')
        new_df['Județ'] = standardize_judet_name(mapped_judet)
    else:
        new_df['Județ'] = 'unknown'
    
    # 4. county_sales_april_2025_log
    if 'county_sales_april_2025_log' in original_cols_order_list:
        # print(f"DEBUG PREDICT: Initial future_raw_county_sales = {future_raw_county_sales}, Județ = {new_df['Județ'].iloc[0]}") # DEBUG
        # Priority 1: Use future_raw_county_sales if provided and valid
        if future_raw_county_sales is not None:
            if pd.notna(future_raw_county_sales) and isinstance(future_raw_county_sales, (int, float)) and future_raw_county_sales >= 0:
                new_df['county_sales_april_2025_log'] = np.log1p(float(future_raw_county_sales))
                # print(f"DEBUG PREDICT: Used future_raw_county_sales. Log value: {new_df['county_sales_april_2025_log'].iloc[0]}") # DEBUG
            else:
                print(f"Warning: Invalid 'future_raw_county_sales' ({future_raw_county_sales}). Using median log sales as fallback.")
                new_df['county_sales_april_2025_log'] = num_medians_dict.get('county_sales_april_2025_log', 0)
                # print(f"DEBUG PREDICT: Fallback to median (invalid future). Log value: {new_df['county_sales_april_2025_log'].iloc[0]}") # DEBUG
        # Priority 2: If no future value, use april_sales_map_lookup based on derived Județ
        elif april_sales_map_lookup: # Check if the map itself is not empty/None
            judet_for_lookup = new_df['Județ'].iloc[0]
            raw_sales_value_for_pred = np.nan
            if judet_for_lookup != 'unknown' and judet_for_lookup in april_sales_map_lookup:
                raw_sales_value_for_pred = april_sales_map_lookup[judet_for_lookup]
            
            if pd.notna(raw_sales_value_for_pred):
                new_df['county_sales_april_2025_log'] = np.log1p(raw_sales_value_for_pred)
                # print(f"DEBUG PREDICT: Used april_sales_map_lookup for Județ {judet_for_lookup}. Log value: {new_df['county_sales_april_2025_log'].iloc[0]}") # DEBUG
            else:
                new_df['county_sales_april_2025_log'] = num_medians_dict.get('county_sales_april_2025_log', 0)
                # print(f"DEBUG PREDICT: Fallback to median (Județ not in map or no raw sales). Log value: {new_df['county_sales_april_2025_log'].iloc[0]}") # DEBUG
        # Priority 3: Fallback to median if no future value and no map lookup possible
        else:
            new_df['county_sales_april_2025_log'] = num_medians_dict.get('county_sales_april_2025_log', 0)
            # print(f"DEBUG PREDICT: Fallback to median (no map). Log value: {new_df['county_sales_april_2025_log'].iloc[0]}") # DEBUG
    
    for col in num_cols_model_list:
        if col not in new_df.columns:
            new_df[col] = num_medians_dict.get(col, 0)
        elif pd.isna(new_df[col].iloc[0]):
            new_df[col] = num_medians_dict.get(col, 0)
    for col_original_name, encoder in encoders_dict.items():
        if col_original_name in new_df.columns:
            new_df[col_original_name] = new_df[col_original_name].astype(str)
            try:
                new_df[col_original_name] = encoder.transform(new_df[col_original_name].astype(str).values.ravel())
            except ValueError as e:
                new_df[col_original_name] = new_df[col_original_name].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)
    final_pred_df = pd.DataFrame(columns=original_cols_order_list)
    for col in original_cols_order_list:
        if col in new_df.columns:
            final_pred_df[col] = new_df[col]
        elif col in num_medians_dict:
            final_pred_df[col] = num_medians_dict[col]
        elif col in encoders_dict:
            final_pred_df[col] = -1
        else:
            final_pred_df[col] = 0
    final_pred_df = final_pred_df[original_cols_order_list]
    try:
        prediction_log = model.predict(final_pred_df)
        prediction_original = np.expm1(prediction_log)
        return prediction_original[0]
    except Exception as e:
        print(f"Error during prediction for new apartment: {e}")
        return None

example_apartment = {
    'address': 'Craiova, Dolj',
    'latitude': 44.3167,
    'longitude': 23.8000,
    'surface': 87, 'rooms': 4, 'floor': '4',
    'construction_year': 1985, 'price_per_sqm': 1675,
    'heating_type': 'centralizată', 'rent': '0',
    'property_status': 'gata de utilizare', 'property_type': 'locuință utilizată',
    'property form': 'drept de proprietate', 'freeFrom': '2025-05-21',
    'sellerType': 'agenție', 'building_type': 'bloc', 'accessibility_label': 'Bună',
    'distance_school': 300, 'has_school_nearby': 1,
    'distance_park': 200, 'has_park_nearby': 1,
    'distance_transport': 50, 'has_transport_nearby': 1,
    'distance_supermarket': 100, 'has_supermarket_nearby': 1,
    'distance_hospital': 800, 'has_hospital_nearby': 1,
    'distance_restaurant': 100, 'has_restaurant_nearby': 1,
    'distance_gym': 400, 'has_gym_nearby': 1,
    'distance_mall': 1000, 'has_mall_nearby': 1,
    'accessibility_score': 90.0, 'facility_count': 8,
}
if 'xgb_reg' in locals():
    predicted_price_example = predict_new_apartment(
        example_apartment, xgb_reg, 
        numerical_cols_model, categorical_cols_model, 
        numerical_medians_model, categorical_modes_model, 
        encoders, original_X_columns, 
        ROMANIAN_CITIES_FOR_EXTRACTION, CITY_TO_COUNTY_MAP, 
        april_sales_map if 'april_sales_map' in locals() else {},
    )
    if predicted_price_example is not None:
        print(f"\nPredicted price for the example apartment (current market conditions): {predicted_price_example:.2f} EUR")
    else:
        print("\nCould not predict price for the example apartment due to errors.")

    # Example for future prediction
    print("\n--- Example: Predicting with a hypothetical future county sales value ---")
    # For the example_apartment (Craiova, Dolj county)
    # Ensure its Județ is correctly identified as 'dolj' by the extract_city and map
    
    # We need to find what april_sales_map.get('dolj') would be if address is correct
    # Let's assume extract_city works and maps example_apartment to 'dolj' county for this example section
    # If it doesn't, the following dolj_current_sales will be based on the actual (potentially wrong) extracted county
    # or a default. The debug logs inside predict_new_apartment will show the truth.

    # For the example, let's get the actual county used for example_apartment's first prediction:
    temp_df_for_county_check = pd.DataFrame([example_apartment])
    temp_df_for_county_check['city'] = temp_df_for_county_check['address'].apply(lambda x: extract_city(x, ROMANIAN_CITIES_FOR_EXTRACTION))
    temp_df_for_county_check['Județ'] = 'unknown' # Default
    if CITY_TO_COUNTY_MAP and 'city' in temp_df_for_county_check.columns:
        city_val = temp_df_for_county_check['city'].iloc[0]
        if city_val != 'Unknown':
             mapped_j = CITY_TO_COUNTY_MAP.get(city_val, 'Unknown')
             temp_df_for_county_check['Județ'] = standardize_judet_name(mapped_j)
    
    actual_county_for_example = temp_df_for_county_check['Județ'].iloc[0]
    # print(f"DEBUG Example Call: Address '{example_apartment.get('address')}' mapped to Județ: {actual_county_for_example}")

    # Use the actual_county_for_example for the baseline sales figure
    current_sales_for_actual_county = april_sales_map.get(actual_county_for_example, 1000) # Default if county not in map
    hypothetical_future_sales = current_sales_for_actual_county * 1.10 # e.g., 10% increase
    
    predicted_price_future_example = predict_new_apartment(
        example_apartment, 
        xgb_reg, 
        numerical_cols_model, categorical_cols_model, 
        numerical_medians_model, categorical_modes_model, 
        encoders, original_X_columns, 
        ROMANIAN_CITIES_FOR_EXTRACTION, CITY_TO_COUNTY_MAP, 
        april_sales_map if 'april_sales_map' in locals() else {}, 
        future_raw_county_sales=hypothetical_future_sales 
    )
    if predicted_price_future_example is not None:
        print(f"For apartment in {example_apartment.get('address', 'N/A')}, originally mapped to Județ '{actual_county_for_example}':")
        print(f"  With current county sales ({current_sales_for_actual_county:.0f}), predicted price: {predicted_price_example:.2f} EUR")
        print(f"  With hypothetical future county sales ({hypothetical_future_sales:.0f}), predicted price: {predicted_price_future_example:.2f} EUR")
    else:
        print("\nCould not predict price for the future scenario example apartment due to errors.")


def evaluate_new_apartment_with_price(apartment_data_with_price, model, 
                                      num_cols_model_list, cat_cols_model_list, 
                                      num_medians_dict, cat_modes_dict, encoders_dict, 
                                      original_cols_order_list, 
                                      city_list_for_extraction, city_to_county_map, 
                                      april_sales_map_lookup,
                                      future_raw_county_sales=None): # Added new param
    print(f"\n--- Evaluating new apartment with known price: ---")
    print(f"Input Address: {apartment_data_with_price.get('address', 'N/A')}, Surface: {apartment_data_with_price.get('surface', 'N/A')}, Actual Price: {apartment_data_with_price.get('price', 'N/A')}")
    if future_raw_county_sales is not None:
        print(f"Using hypothetical future raw county sales for evaluation: {future_raw_county_sales}")

    if 'price' not in apartment_data_with_price:
        print("Error: 'price' key is missing from apartment_data_with_price. Cannot evaluate.")
        return
    actual_price = apartment_data_with_price.get('price')
    try:
        actual_price = float(actual_price)
    except (ValueError, TypeError):
        print(f"Error: Provided actual price '{actual_price}' is not a valid number.")
        return
    apartment_features_for_prediction = {k: v for k, v in apartment_data_with_price.items() if k != 'price'}
    predicted_price = predict_new_apartment(
        apartment_features_for_prediction, model, 
        num_cols_model_list, cat_cols_model_list, 
        num_medians_dict, cat_modes_dict, encoders_dict, 
        original_cols_order_list, city_list_for_extraction, 
        city_to_county_map, april_sales_map_lookup,
        future_raw_county_sales=future_raw_county_sales # Pass it through
    )
    if predicted_price is not None:
        residual = actual_price - predicted_price
        print(f"  Actual Price:    {actual_price:.2f} EUR")
        print(f"  Predicted Price: {predicted_price:.2f} EUR")
        print(f"  Residual:        {residual:.2f} EUR")
    else:
        print("\nCould not get a prediction to evaluate against the actual price.")

example_apartment_with_known_price = {
    'price': 115000,
    'address': 'Calea Victoriei 100, Bucuresti, Sector 1',
    'latitude': 44.4350, 'longitude': 26.0990,
    'surface': 65, 'rooms': 2, 'floor': '3',
    'construction_year': 1990, 'price_per_sqm': 1769,
    'heating_type': 'centrală pe gaz', 'rent': '450',
    'property_status': 'gata de utilizare', 'property_type': 'locuință utilizată',
    'property form': 'drept de proprietate', 'freeFrom': '2025-05-21',
    'sellerType': 'agenție', 'building_type': 'bloc', 'accessibility_label': 'Foarte Bună',
    'distance_school': 250, 'has_school_nearby': 1,
    'distance_park': 150, 'has_park_nearby': 1,
    'distance_transport': 100, 'has_transport_nearby': 1,
    'distance_supermarket': 200, 'has_supermarket_nearby': 1,
    'distance_hospital': 700, 'has_hospital_nearby': 1,
    'distance_restaurant': 50, 'has_restaurant_nearby': 1,
    'distance_gym': 300, 'has_gym_nearby': 1,
    'distance_mall': 1200, 'has_mall_nearby': 1,
    'accessibility_score': 95.0, 'facility_count': 8
}
if 'xgb_reg' in locals():
    evaluate_new_apartment_with_price(
        example_apartment_with_known_price, xgb_reg, 
        numerical_cols_model, categorical_cols_model, 
        numerical_medians_model, categorical_modes_model, 
        encoders, original_X_columns, 
        ROMANIAN_CITIES_FOR_EXTRACTION, CITY_TO_COUNTY_MAP, 
        april_sales_map if 'april_sales_map' in locals() else {}
        # future_raw_county_sales is None for standard evaluation
    )

    # Example of evaluating with a future sales value
    print("\n--- Example: Evaluating with known price AND a hypothetical future county sales value ---")
    # Assuming Bucharest sales might increase by 5%
    bucuresti_current_sales = april_sales_map.get('bucuresti', 5000) # Get current or default
    hypothetical_future_bucuresti_sales = bucuresti_current_sales * 1.05 

    evaluate_new_apartment_with_price(
        example_apartment_with_known_price, # Using the same Bucharest apartment
        xgb_reg, 
        numerical_cols_model, categorical_cols_model, 
        numerical_medians_model, categorical_modes_model, 
        encoders, original_X_columns, 
        ROMANIAN_CITIES_FOR_EXTRACTION, CITY_TO_COUNTY_MAP, 
        april_sales_map if 'april_sales_map' in locals() else {},
        future_raw_county_sales=hypothetical_future_bucuresti_sales
    )


print("\n\nBenchmark script finished.")
