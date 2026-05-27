"""
Converts raw dataset files from wide format to long format
train/test CSVs. Run this before any model scripts.
Place raw data files in the same directory before running.

Inputs (place alongside this script):
  M4         : Daily-train.csv, Daily-test.csv, M4-info.csv
  Traffic    : traffic.csv
  Exchange   : exchange_rate.csv
  ETTh1      : ETTh1.csv

Outputs (written to current directory):
  M4         : M4_Daily_Train_PLX.csv, M4_Daily_Test_PLX.csv
  Traffic    : Traffic_Daily_Train.csv, Traffic_Daily_Test.csv
  Exchange   : Exchange_Rate_Train.csv, Exchange_Rate_Test.csv
  ETTh1      : ETTh1_Train.csv, ETTh1_Test.csv
"""

import os
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# M4 Daily
# ---------------------------------------------------------------------------
def process_m4_data_fixed():
    TRAIN_FILE = 'Daily-train.csv'
    TEST_FILE  = 'Daily-test.csv'
    INFO_FILE  = 'M4-info.csv'

    print("1. Loading Data...")
    df_train = pd.read_csv(TRAIN_FILE)
    df_test  = pd.read_csv(TEST_FILE)
    df_info  = pd.read_csv(INFO_FILE)

    # Standardize ID columns
    df_train.rename(columns={'V1': 'series_id'}, inplace=True)
    df_test.rename(columns={'V1': 'series_id'}, inplace=True)
    df_info.rename(columns={'M4id': 'series_id'}, inplace=True)

    print(f"   Train Shape: {df_train.shape}")
    print(f"   Test Shape: {df_test.shape}")

    # --- STEP 2: ROBUST MELT (Skip complex metadata logic) ---
    print("2. Melting Training Data...")

    # Identify value columns (V2, V3...)
    train_value_cols = [c for c in df_train.columns if c.startswith('V')]

    df_train_long = df_train.melt(
        id_vars=['series_id'],
        value_vars=train_value_cols,
        var_name='col_code',
        value_name='sales'
    )

    # Drop NaNs (removes the empty tail of the wide matrix)
    df_train_long.dropna(subset=['sales'], inplace=True)

    # Calculate Day Offset (V2 -> 0, V3 -> 1...)
    df_train_long['day_offset'] = (
        df_train_long['col_code'].str.extract(r'(\d+)').astype(int) - 2
    )

    # Join with Start Date from Info file
    print("   Mapping Dates...")
    df_train_final = df_train_long.merge(
        df_info[['series_id', 'StartingDate']],
        on='series_id', how='left'
    )
    df_train_final['start_date'] = pd.to_datetime(df_train_final['StartingDate'])

    df_train_final['date'] = (
        df_train_final['start_date']
        + pd.to_timedelta(df_train_final['day_offset'], unit='D')
    )

    df_train_output = (df_train_final[['series_id', 'date', 'sales']]
                       .sort_values(['series_id', 'date']))
    df_train_output['split_type'] = 'TRAIN'

    print(f"   Saving M4_Daily_Train_PLX.csv ({len(df_train_output)} rows)...")
    df_train_output.to_csv('M4_Daily_Train_PLX.csv', index=False)

    # --- STEP 3: PROCESS TEST DATA ---
    print("3. Melting Test Data...")

    test_value_cols = [c for c in df_test.columns if c.startswith('V')]

    df_test_long = df_test.melt(
        id_vars=['series_id'],
        value_vars=test_value_cols,
        var_name='col_code',
        value_name='sales'
    )
    df_test_long.dropna(subset=['sales'], inplace=True)

    # Need the end of training to know where Test starts
    train_end_dates = (df_train_output.groupby('series_id')['date']
                       .max().reset_index())
    train_end_dates.rename(columns={'date': 'train_end_date'}, inplace=True)

    df_test_final = df_test_long.merge(train_end_dates, on='series_id', how='left')

    # Extract test offset (V2 -> 0, V3 -> 1...)
    df_test_final['test_step'] = (
        df_test_final['col_code'].str.extract(r'(\d+)').astype(int) - 2
    )

    # Actual Test Date: Train End + 1 Day + Step Offset
    df_test_final['date'] = (
        df_test_final['train_end_date']
        + pd.to_timedelta(df_test_final['test_step'] + 1, unit='D')
    )

    df_test_output = (df_test_final[['series_id', 'date', 'sales']]
                      .sort_values(['series_id', 'date']))
    df_test_output['split_type'] = 'TEST'

    print(f"   Saving M4_Daily_Test_PLX.csv ({len(df_test_output)} rows)...")
    df_test_output.to_csv('M4_Daily_Test_PLX.csv', index=False)

    print("\n✅ SUCCESS! M4 Data Reprocessed correctly.")
    print(f"Train Row Count: {len(df_train_output)}")
    print(f"Test Row Count:  {len(df_test_output)}")


# ---------------------------------------------------------------------------
# Shared melt helper for wide → long datasets with a 'date' column
# ---------------------------------------------------------------------------
def _melt_wide(df_wide, split_name):
    series_cols = [c for c in df_wide.columns if c != 'date']
    df_long = df_wide.melt(
        id_vars=['date'],
        value_vars=series_cols,
        var_name='series_id',
        value_name='value'
    )
    df_long = df_long.sort_values(['series_id', 'date'])
    df_long['split_type'] = split_name
    return df_long


# ---------------------------------------------------------------------------
# Traffic (PeMS)
# ---------------------------------------------------------------------------
def process_traffic_data():
    INPUT_FILE   = 'traffic.csv'
    TRAIN_OUTPUT = 'Traffic_Daily_Train.csv'
    TEST_OUTPUT  = 'Traffic_Daily_Test.csv'

    print("1. Loading Data...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        print("   Note: 'date' column not found, assuming first column is timestamp.")
        date_col_name = df.columns[0]
        df.rename(columns={date_col_name: 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])

    print(f"   Total Data Shape: {df.shape}")

    # --- STEP 2: SPLIT (TIME BASED, last 20% for test) ---
    print("2. Splitting Train/Test (Last 20% for Test)...")
    total_rows  = len(df)
    split_index = int(total_rows * 0.8)

    df_train_wide = df.iloc[:split_index].copy()
    df_test_wide  = df.iloc[split_index:].copy()

    print(f"   Train Rows (Wide): {len(df_train_wide)}")
    print(f"   Test Rows (Wide):  {len(df_test_wide)}")

    # --- STEP 3: MELT ---
    df_train_final = _melt_wide(df_train_wide, 'TRAIN')
    df_test_final  = _melt_wide(df_test_wide,  'TEST')

    # --- STEP 4: SAVE ---
    print("3. Saving Output Files...")
    print(f"   Saving {TRAIN_OUTPUT} ({len(df_train_final)} rows)...")
    df_train_final.to_csv(TRAIN_OUTPUT, index=False)
    print(f"   Saving {TEST_OUTPUT} ({len(df_test_final)} rows)...")
    df_test_final.to_csv(TEST_OUTPUT, index=False)

    print("\n✅ SUCCESS! Traffic Data Reprocessed correctly.")


# ---------------------------------------------------------------------------
# Exchange Rate
# ---------------------------------------------------------------------------
def process_exchange_rate_data():
    INPUT_FILE   = 'exchange_rate.csv'
    TRAIN_OUTPUT = 'Exchange_Rate_Train.csv'
    TEST_OUTPUT  = 'Exchange_Rate_Test.csv'

    print("1. Loading Data...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # --- DATE HANDLING ---
    date_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'timestamp' in col.lower():
            date_col = col
            break

    if date_col:
        print(f"   Detected date column: {date_col}")
        df['date'] = pd.to_datetime(df[date_col])
        if date_col != 'date':
            df.drop(columns=[date_col], inplace=True)
    else:
        print("   No date column found. Generating daily index starting 1990-01-01...")
        df['date'] = pd.date_range(start='1990-01-01', periods=len(df), freq='D')

    print(f"   Total Data Shape: {df.shape}")

    # --- STEP 2: SPLIT ---
    print("2. Splitting Train/Test (Last 20% for Test)...")
    total_rows  = len(df)
    split_index = int(total_rows * 0.8)

    df_train_wide = df.iloc[:split_index].copy()
    df_test_wide  = df.iloc[split_index:].copy()

    print(f"   Train Rows (Wide): {len(df_train_wide)}")
    print(f"   Test Rows (Wide):  {len(df_test_wide)}")

    # --- STEP 3: MELT ---
    df_train_final = _melt_wide(df_train_wide, 'TRAIN')
    df_test_final  = _melt_wide(df_test_wide,  'TEST')

    # --- STEP 4: SAVE ---
    print("3. Saving Output Files...")
    print(f"   Saving {TRAIN_OUTPUT} ({len(df_train_final)} rows)...")
    df_train_final.to_csv(TRAIN_OUTPUT, index=False)
    print(f"   Saving {TEST_OUTPUT} ({len(df_test_final)} rows)...")
    df_test_final.to_csv(TEST_OUTPUT, index=False)

    print("\n✅ SUCCESS! Exchange Rate Data Reprocessed correctly.")


# ---------------------------------------------------------------------------
# ETTh1
# ---------------------------------------------------------------------------
def process_etth1_data():
    INPUT_FILE   = 'ETTh1.csv'
    TRAIN_OUTPUT = 'ETTh1_Train.csv'
    TEST_OUTPUT  = 'ETTh1_Test.csv'

    print("1. Loading Data...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please upload ETTh1.csv")
        return

    df = pd.read_csv(INPUT_FILE)

    # --- DATE HANDLING ---
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        print("   Warning: 'date' column not found. Using first column...")
        date_col = df.columns[0]
        df.rename(columns={date_col: 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])

    print(f"   Total Data Shape: {df.shape}")

    # --- STEP 2: SPLIT ---
    print("2. Splitting Train/Test (Last 20% for Test)...")
    total_rows  = len(df)
    split_index = int(total_rows * 0.8)

    df_train_wide = df.iloc[:split_index].copy()
    df_test_wide  = df.iloc[split_index:].copy()

    print(f"   Train Rows (Wide): {len(df_train_wide)}")
    print(f"   Test Rows (Wide):  {len(df_test_wide)}")

    # --- STEP 3: MELT ---
    df_train_final = _melt_wide(df_train_wide, 'TRAIN')
    df_test_final  = _melt_wide(df_test_wide,  'TEST')

    # --- STEP 4: SAVE ---
    print("3. Saving Output Files...")
    print(f"   Saving {TRAIN_OUTPUT} ({len(df_train_final)} rows)...")
    df_train_final.to_csv(TRAIN_OUTPUT, index=False)
    print(f"   Saving {TEST_OUTPUT} ({len(df_test_final)} rows)...")
    df_test_final.to_csv(TEST_OUTPUT, index=False)

    print("\n✅ SUCCESS! ETTh1 Data Reprocessed correctly.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    process_m4_data_fixed()
    process_traffic_data()
    process_exchange_rate_data()
    process_etth1_data()
