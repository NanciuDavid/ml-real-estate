import pandas as pd
import os
import re
from datetime import datetime

ANCPI_DATA_DIR = 'ancpi statistica tranzactii'
OUTPUT_TREND_FILE = 'market_trends_ancpi.csv'

def parse_filename(filename):
    """
    Parses the ANCPI filename to extract month, year, and data type.
    Example: "Aprilie-vanzari-2025.xlsx" -> (4, 2025, "vanzari")
    """
    month_map = {
        'ianuarie': 1, 'februarie': 2, 'martie': 3, 'aprilie': 4, 'mai': 5, 'iunie': 6,
        'iulie': 7, 'august': 8, 'septembrie': 9, 'octombrie': 10, 'noiembrie': 11, 'decembrie': 12
    }
    
    # Regex to capture month, type, and year
    # It handles variations like "Decembrie_-cereri-2024.xlsx" or "Noiembrie_ipoteci-2024.xlsx"
    match = re.match(r"([A-Za-z]+)[\s_-]*(cereri|ipoteci|vanzari)[\s_-]*(\d{4})\.xlsx", filename, re.IGNORECASE)
    
    if match:
        month_str, type_str, year_str = match.groups()
        month = month_map.get(month_str.lower())
        year = int(year_str)
        data_type = type_str.lower()
        
        if month:
            return month, year, data_type
    return None, None, None

def load_and_inspect_ancpi_file(filepath, sheet_name_to_load=None, header_row=0):
    """
    Loads a specific sheet from an Excel file.
    Inspects its columns and head.

    Args:
        filepath (str): Path to the Excel file.
        sheet_name_to_load (str, optional): The specific sheet name to load. Defaults to None.
        header_row (int, optional): The row to use for headers (0-indexed). Defaults to 0.

    Returns:
        tuple: (pandas.DataFrame or None, str or None) - The loaded DataFrame and the sheet name, or (None, None) if loading fails.
    """
    try:
        xls = pd.ExcelFile(filepath)
        sheet_names = xls.sheet_names
        # print(f"  Sheets found in {os.path.basename(filepath)}: {sheet_names}")

        current_sheet_to_try = None
        df_temp = None
        header_row_to_use = header_row # Directly use the provided header_row

        if sheet_name_to_load and sheet_name_to_load in sheet_names:
            current_sheet_to_try = sheet_name_to_load
            print(f"  Attempting to load specified sheet: '{current_sheet_to_try}' with header on row {header_row_to_use}")
            df_temp = pd.read_excel(filepath, sheet_name=current_sheet_to_try, header=header_row_to_use)
        elif not sheet_name_to_load and sheet_names: # If no sheet specified, try the first one
            current_sheet_to_try = sheet_names[0]
            print(f"  No sheet specified, attempting to load first sheet: '{current_sheet_to_try}' with header on row {header_row_to_use}")
            df_temp = pd.read_excel(filepath, sheet_name=current_sheet_to_try, header=header_row_to_use)
        else:
            if not sheet_names:
                print(f"Error: No sheets found in {filepath}.")
            else:
                print(f"Error: Sheet '{sheet_name_to_load}' not found in {filepath}. Available: {sheet_names}")
            return None, None

        if df_temp is None:
            print(f"  Could not load any sheet for {filepath}. Skipping.")
            return None, None
        
        # --- DEBUG: Print head of DataFrame after initial load ---\n        print(f\"  DEBUG: Head of sheet \'{current_sheet_to_try}\' with header={header_row_to_use}:\")
        print(f"  DEBUG: Head of sheet '{current_sheet_to_try}' with header={header_row_to_use}:")
        print(df_temp.head())
        print(f"  DEBUG: Columns detected by pandas: {df_temp.columns.tolist()}")
        print(f"  DEBUG: Shape: {df_temp.shape}")
        return df_temp, current_sheet_to_try
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None
    except Exception as e:
        print(f"Error reading Excel file {filepath} (sheet: {sheet_name_to_load}, header: {header_row_to_use}): {e}")
        return None, None

def process_all_ancpi_files():
    """
    Processes ANCPI files, focusing on 'vanzari' type for now.
    """
    all_files = [f for f in os.listdir(ANCPI_DATA_DIR) if f.endswith('.xlsx') and not f.startswith('~')]
    if not all_files:
        print(f"No Excel files found in {ANCPI_DATA_DIR}")
        return
        
    master_data_list = []

    for filename in all_files:
        month, year, data_type = parse_filename(filename)
        if not month or not year or not data_type:
            print(f"Could not parse filename: {filename}. Skipping.")
            continue
        
        # --- FOCUS ONLY ON VANZARI FILES FOR NOW ---
        if data_type != 'vanzari':
            print(f"Skipping non-vanzari file: {filename}")
            continue
        
        filepath = os.path.join(ANCPI_DATA_DIR, filename)
        print(f"Processing: {filepath} for {year}-{month:02d} ({data_type})")
        
        filename_lower = filename.lower()
        if "vanzari" in filename_lower:
            df, sheet_name = load_and_inspect_ancpi_file(
                filepath, 
                sheet_name_to_load="VANZARI" 
                # header_row will be handled by the function for VANZARI sheets
            )
            if df is not None and sheet_name is not None:
                assumed_sheet_name = 'VANZARI' # Specific for sales files
                metric_col_name = 'Total Imobile'
                # rows_to_skip = 1 # We will use header parameter directly

                current_sheet_to_try = assumed_sheet_name
                df_temp = None

                try:
                    try:
                        # Use header=1 to specify that the second row (index 1) in Excel contains the headers
                        df_temp = pd.read_excel(filepath, sheet_name=current_sheet_to_try, header=2)
                    except ValueError as e:
                        if "Worksheet named" in str(e) and "not found" in str(e):
                            print(f"  Sheet '{current_sheet_to_try}' not found for {filename}. Attempting to use first available sheet.")
                            xls_sheets = pd.ExcelFile(filepath).sheet_names
                            if xls_sheets:
                                current_sheet_to_try = xls_sheets[0]
                                print(f"  Trying sheet: '{current_sheet_to_try}'")
                                # Use header=1 for the fallback sheet as well
                                df_temp = pd.read_excel(filepath, sheet_name=current_sheet_to_try, header=2)
                            else:
                                print(f"  No sheets found in {filename}. Skipping.")
                                continue
                        else:
                            raise # Re-raise other ValueErrors
                    
                    if df_temp is None:
                        print(f"  Could not load any sheet for {filename}. Skipping.")
                        continue
                    
                    # --- DEBUG: Print head of DataFrame after initial load ---
                    print(f"  DEBUG: Head of sheet '{current_sheet_to_try}' with header=1:")
                    print(df_temp.head(5))
                    print(f"  DEBUG: Columns detected by pandas: {df_temp.columns.tolist()}")

                    # Standardize column names by stripping whitespace
                    df_temp.columns = [str(col).strip() for col in df_temp.columns]

                    judet_col_found = None
                    if 'Județ' in df_temp.columns:
                        judet_col_found = 'Județ'
                    elif 'Judet' in df_temp.columns:
                        judet_col_found = 'Judet'
                    
                    if judet_col_found:
                        df_temp.dropna(subset=[judet_col_found], inplace=True)
                        df_temp = df_temp[df_temp[judet_col_found].astype(str).str.upper() != 'TOTAL']
                    else:
                        print(f"  Column for County ('Județ' or 'Judet') not found in {filename} (sheet: {current_sheet_to_try}). Columns: {df_temp.columns.tolist()}")
                        continue # Skip this file if county column is not found

                    actual_metric_col_to_use = None
                    if metric_col_name in df_temp.columns:
                        actual_metric_col_to_use = metric_col_name
                    elif 'Numar Imobile' in df_temp.columns: 
                        actual_metric_col_to_use = 'Numar Imobile'
                        print(f"  Used metric column 'Numar Imobile' for {filename} (original guess: '{metric_col_name}')")
                    elif 'Număr Imobile' in df_temp.columns:
                         actual_metric_col_to_use = 'Număr Imobile'
                         print(f"  Used metric column 'Număr Imobile' for {filename} (original guess: '{metric_col_name}')")
                    elif 'TOTAL' in df_temp.columns: 
                        actual_metric_col_to_use = 'TOTAL'
                        print(f"  Used fallback metric column 'TOTAL' for {filename} (original guess: '{metric_col_name}')")
                    elif metric_col_name.upper() in df_temp.columns:
                         actual_metric_col_to_use = metric_col_name.upper()
                         print(f"  Used ALL CAPS metric column '{actual_metric_col_to_use}' for {filename}")
                    else: 
                        possible_total_cols = [col for col in df_temp.columns if 'TOTAL' in str(col).upper()]
                        if possible_total_cols:
                            actual_metric_col_to_use = possible_total_cols[0]
                            print(f"  Used auto-detected total column '{actual_metric_col_to_use}' for {filename}")
                        else: 
                            print(f"  Metric column '{metric_col_name}' (or fallbacks) not found in {filename} (sheet: {current_sheet_to_try}). Columns: {df_temp.columns.tolist()}")
                            continue
                    
                    df_extracted = df_temp[[judet_col_found, actual_metric_col_to_use]].copy()
                    df_extracted.rename(columns={judet_col_found: 'Judet', actual_metric_col_to_use: 'Valoare'}, inplace=True)
                    
                    df_extracted['An'] = year
                    df_extracted['Luna'] = month
                    df_extracted['TipDate'] = data_type # This will be 'vanzari'
                    
                    df_extracted['Valoare'] = pd.to_numeric(df_extracted['Valoare'], errors='coerce')
                    df_extracted.dropna(subset=['Valoare'], inplace=True)

                    master_data_list.append(df_extracted)
                    print(f"  Successfully processed and extracted data from {filename}. Shape: {df_extracted.shape}")

                except Exception as e:
                    print(f"  Major error processing {filename} (sheet: {current_sheet_to_try}): {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
        # Temporarily skip ipoteci and cereri
        # elif "ipoteci" in filename_lower:
        #     df, sheet_name = load_and_inspect_ancpi_file(filepath, sheet_name_to_load="DINAMICA IPOTECI", header_row=1) 
        # elif "cereri" in filename_lower:
        #     df, sheet_name = load_and_inspect_ancpi_file(filepath, sheet_name_to_load="DINAMICA CERERI", header_row=1) 
        else:
            print(f"Skipping non-vanzari file: {filename}")
            continue
            
    if not master_data_list:
        print("No VANZARI data was successfully extracted from any files. Exiting.")
        return

    df_aggregated = pd.concat(master_data_list, ignore_index=True)
    print("\n--- Aggregated VANZARI Data --- ")
    print(f"Shape: {df_aggregated.shape}")
    print(df_aggregated.head())
    print("\nValue counts for TipDate:")
    print(df_aggregated['TipDate'].value_counts())
    print("\nData types:")
    print(df_aggregated.dtypes)
    print("\nMissing values per column:")
    print(df_aggregated.isnull().sum())

    # Pivot table to have one row per An-Luna-Judet, and columns for each TipDate value
    df_pivot = df_aggregated.pivot_table(
        index=['An', 'Luna', 'Judet'],
        columns='TipDate',
        values='Valoare'
    ).reset_index()
    
    df_pivot.columns = [f'{col}' if col in ['An', 'Luna', 'Judet'] else f'Valoare_{col}' for col in df_pivot.columns]
    
    print("\n--- Pivoted VANZARI Data --- ")
    print(f"Shape: {df_pivot.shape}")
    print(df_pivot.head())
    print("\nMissing values in pivoted VANZARI data:")
    print(df_pivot.isnull().sum())

    df_pivot = df_pivot.sort_values(by=['Judet', 'An', 'Luna']).reset_index(drop=True)
    
    value_cols = [col for col in df_pivot.columns if col.startswith('Valoare_')]
    for val_col in value_cols:
        df_pivot[f'{val_col}_lag_1m'] = df_pivot.groupby('Judet')[val_col].shift(1)

    print("\n--- Pivoted VANZARI Data with Lags --- ")
    print(f"Shape: {df_pivot.shape}")
    print(df_pivot.head(10))
    print("\nMissing values after lag generation:")
    print(df_pivot.isnull().sum())

    # Save only vanzari data for now
    vanzari_output_file = 'market_trends_vanzari_ancpi.csv'
    df_pivot.to_csv(vanzari_output_file, index=False)
    print(f"\nSuccessfully saved processed ANCPI VANZARI data to {vanzari_output_file}")

def process_consolidated_sales_file():
    """
    Processes the consolidated ANCPI sales data from the single Excel file.
    """
    consolidated_filename = "Real estate sales from nov 2024-apr 2025.xlsx"
    output_csv_filename = "market_trends_vanzari_ancpi.csv"

    print(f"Starting processing of consolidated file: {consolidated_filename}")

    # Step 1: Load the first row to get the actual month-year column headers
    try:
        df_first_row = pd.read_excel(consolidated_filename, sheet_name='Sheet1', header=None, nrows=1)
        month_year_headers = df_first_row.iloc[0].tolist() # Get as a list
        print(f"Successfully loaded first row for headers: {month_year_headers}")
    except Exception as e:
        print(f"Could not load the first row for headers from {consolidated_filename}: {e}")
        return

    # Step 2: Load the main data, using the second row (index 1) as the primary header source
    df_main_data, sheet_name = load_and_inspect_ancpi_file(
        consolidated_filename, 
        sheet_name_to_load='Sheet1', 
        header_row=1 # Actual data headers are on the second row (index 1)
    )

    if df_main_data is None:
        print(f"Could not load main data from {consolidated_filename}. Exiting.")
        return

    print(f"Successfully loaded sheet '{sheet_name}' for main data. Initial columns from load_and_inspect: {df_main_data.columns.tolist()}")

    # Step 3: Rename data columns using the headers from the first row
    # The first few columns from header_row=1 load ('Nr. crt.', 'Județ') should be preserved.
    # The rest of the columns should be named from month_year_headers.
    num_fixed_cols = 0
    if 'Nr. crt.' in df_main_data.columns: num_fixed_cols +=1
    if 'Județ' in df_main_data.columns: num_fixed_cols +=1
    
    new_column_names = df_main_data.columns[:num_fixed_cols].tolist() + month_year_headers[num_fixed_cols:]
    
    # Ensure the number of new column names matches the DataFrame's columns
    if len(new_column_names) == len(df_main_data.columns):
        df_main_data.columns = new_column_names
        print(f"Renamed columns to: {df_main_data.columns.tolist()}")
    else:
        print(f"Error: Mismatch in column count for renaming. Expected {len(df_main_data.columns)}, got {len(new_column_names)} from first row headers.")
        print(f"Original columns: {df_main_data.columns.tolist()}")
        print(f"Headers from first row: {month_year_headers}")
        return

    df = df_main_data # Use df for further processing as per original logic

    # Drop rows that are entirely NaN (often occur between header and data)
    df.dropna(how='all', inplace=True)
    # Drop rows where 'Județ' is NaN (if such a column exists after header loading)
    if 'Județ' in df.columns:
        df.dropna(subset=['Județ'], inplace=True)
        # Filter out any summary rows like 'TOTAL' in the Județ column
        df = df[df['Județ'].astype(str).str.upper() != 'TOTAL']
    elif 'Unnamed: 1' in df.columns and df['Unnamed: 1'].iloc[0].strip().lower() == 'județ':
        # If header was misaligned and Județ is in data part of Unnamed: 1
        # This case should ideally be handled by correct header_row, but as a fallback:
        print("Warning: 'Județ' column not found directly, attempting to infer from 'Unnamed: 1'.")
        df.rename(columns={'Unnamed: 1': 'Județ'}, inplace=True)
        df.dropna(subset=['Județ'], inplace=True)
        df = df[df['Județ'].astype(str).str.upper() != 'TOTAL']
    else:
        print("Error: 'Județ' column not found. Cannot proceed with unpivoting.")
        print(f"Columns available: {df.columns.tolist()}")
        return

    # Identify id_vars (columns to keep fixed) and value_vars (columns to unpivot)
    id_vars = ['Județ']
    if 'Nr. crt.' in df.columns: # Keep Nr. crt. if it exists
        id_vars.insert(0, 'Nr. crt.')
    
    value_vars = [col for col in df.columns if col not in id_vars]
    
    print(f"Identified id_vars: {id_vars}")
    print(f"Identified value_vars (to be unpivoted): {value_vars}")

    if not value_vars:
        print("Error: No columns found to unpivot (monthly data columns).")
        return

    # Melt the DataFrame
    df_melted = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='LunaAn',  # This column will contain strings like "Martie 2025"
        value_name='Valoare_vanzari'
    )

    print("DataFrame after melting:")
    print(df_melted.head())

    # --- Parse 'LunaAn' into separate 'Luna' and 'An' columns ---
    month_map_ro_to_int = {
        'ianuarie': 1, 'februarie': 2, 'martie': 3, 'aprilie': 4, 'mai': 5, 'iunie': 6,
        'iulie': 7, 'august': 8, 'septembrie': 9, 'octombrie': 10, 'noiembrie': 11, 'decembrie': 12
    }

    def parse_luna_an(luna_an_str):
        if pd.isna(luna_an_str):
            return None, None
        parts = str(luna_an_str).strip().split()
        if len(parts) == 2:
            month_str, year_str = parts[0], parts[1]
            try:
                year = int(year_str)
                month = month_map_ro_to_int.get(month_str.lower())
                if month:
                    return month, year
            except ValueError:
                pass # Year could not be converted to int
        # Fallback for cases like "Noiembrie\n 2024"
        if '\n' in str(luna_an_str):
            parts = str(luna_an_str).replace('\n', ' ').strip().split()
            if len(parts) == 2:
                month_str, year_str = parts[0], parts[1]
                try:
                    year = int(year_str)
                    month = month_map_ro_to_int.get(month_str.lower())
                    if month:
                        return month, year
                except ValueError:
                    pass 
        print(f"Could not parse LunaAn: '{luna_an_str}'")
        return None, None

    df_melted[['Luna', 'An']] = df_melted['LunaAn'].apply(lambda x: pd.Series(parse_luna_an(x)))

    # Drop rows where Luna or An could not be parsed
    df_melted.dropna(subset=['Luna', 'An'], inplace=True)
    df_melted['Luna'] = df_melted['Luna'].astype(int)
    df_melted['An'] = df_melted['An'].astype(int)
    
    # Convert 'Valoare_vanzari' to numeric, coercing errors
    df_melted['Valoare_vanzari'] = pd.to_numeric(df_melted['Valoare_vanzari'], errors='coerce')
    df_melted.dropna(subset=['Valoare_vanzari'], inplace=True) # Remove rows where conversion failed

    # Select and reorder columns
    final_df = df_melted[['An', 'Luna', 'Județ', 'Valoare_vanzari']]
    final_df = final_df.sort_values(by=['Județ', 'An', 'Luna']).reset_index(drop=True)

    print("\n--- Final Processed Data ---")
    print(f"Shape: {final_df.shape}")
    print(final_df.head())
    print("\nMissing values:")
    print(final_df.isnull().sum())
    print("\nData types:")
    print(final_df.dtypes)

    final_df.to_csv(output_csv_filename, index=False)
    print(f"\nSuccessfully saved processed data to {output_csv_filename}")

if __name__ == '__main__':
    # print(f"Starting ANCPI data processing. Looking in: {os.path.abspath(ANCPI_DATA_DIR)}")
    # process_all_ancpi_files() 
    process_consolidated_sales_file() # Call the new function 