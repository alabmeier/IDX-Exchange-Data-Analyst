import os
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"

# Input files (combined datasets)
LISTINGS_FILE = "listings.csv"
SOLD_FILE = "sold.csv"


# Which date column defines the month of each record
LISTINGS_DATE_COL = "ListingContractDate"
SOLD_DATE_COL = "CloseDate"



def header(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def add_year_month(df, date_col):
    """Create a year_month key (e.g. 2025-03) from a date column."""
    df = df.copy()
    df["year_month"] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M")
    return df


def merge_and_validate(df, mortgage_monthly, name, date_col):
    """Left-merge monthly rates onto df and check that every row got a rate."""
    rows_before = len(df)
    merged = df.merge(mortgage_monthly, on="year_month", how="left")

    header(f"{name} - MERGE VALIDATION")

    print(f"Rows before merge: {rows_before:,}")
    print(f"Rows after merge:  {len(merged):,}")
    if len(merged) != rows_before:
        print("WARNING: row count changed. Check mortgage_monthly for duplicate months.")

    null_rates = merged["rate_30yr_fixed"].isnull().sum()
    print(f"\nNull rate values: {null_rates:,}")

    if null_rates:
        missing_date = merged["year_month"].isnull().sum()
        unmatched_month = null_rates - missing_date
        print(f"  - rows with missing/unparseable {date_col}: {missing_date:,}")
        print(f"  - rows whose month is not in the FRED data: {unmatched_month:,}")
        if unmatched_month:
            months = merged.loc[
                merged["rate_30yr_fixed"].isnull() & merged["year_month"].notnull(),
                "year_month",
            ].value_counts().sort_index()
            print("  Unmatched months:")
            print(months)
        print(f"RESULT: FAIL - {null_rates:,} null rate values remain")
    else:
        print("RESULT: PASS - no null rate values after merge")

    return merged


# =============================================================================
# 1. LOAD MLS DATA
# =============================================================================
folder_path = input("Enter the path to the folder containing the CSV files: ")

listings = pd.read_csv(os.path.join(folder_path, LISTINGS_FILE), low_memory=False)
sold = pd.read_csv(os.path.join(folder_path, SOLD_FILE), low_memory=False)


header("MLS DATA LOADED")
print(f"Listings rows: {len(listings):,}")
print(f"Sold rows:     {len(sold):,}")


# =============================================================================
# 2. FETCH MORTGAGE RATES FROM FRED
# =============================================================================
header("FETCHING FRED MORTGAGE30US")

try:
    mortgage = pd.read_csv(FRED_URL)
except Exception as e:
    raise SystemExit(f"Could not fetch FRED data: {e}\n"
                     "Check your internet connection, or download the CSV manually from "
                     "https://fred.stlouisfed.org/series/MORTGAGE30US and point FRED_URL at the file.")


mortgage = mortgage.iloc[:, :2]
mortgage.columns = ["date", "rate_30yr_fixed"]
mortgage["date"] = pd.to_datetime(mortgage["date"], errors="coerce")
mortgage["rate_30yr_fixed"] = pd.to_numeric(mortgage["rate_30yr_fixed"], errors="coerce")
mortgage = mortgage.dropna()

print(f"Weekly observations: {len(mortgage):,}")
print(f"Date range: {mortgage['date'].min().date()} to {mortgage['date'].max().date()}")
print(mortgage.tail())

# =============================================================================
# 3. RESAMPLE WEEKLY -> MONTHLY AVERAGE
# =============================================================================
mortgage["year_month"] = mortgage["date"].dt.to_period("M")

mortgage_monthly = (
    mortgage.groupby("year_month")["rate_30yr_fixed"]
    .mean()
    .reset_index()
)

header("MONTHLY AVERAGE RATES")
print(f"Months: {len(mortgage_monthly):,}")
print(mortgage_monthly.tail(12))

# =============================================================================
# 4. CREATE year_month KEYS ON THE MLS DATASETS
# =============================================================================
listings = add_year_month(listings, LISTINGS_DATE_COL)
sold = add_year_month(sold, SOLD_DATE_COL)


header("YEAR-MONTH KEY RANGE")
print(f"Listings ({LISTINGS_DATE_COL}): {listings['year_month'].min()} to {listings['year_month'].max()}")
print(f"Sold ({SOLD_DATE_COL}):         {sold['year_month'].min()} to {sold['year_month'].max()}")


# =============================================================================
# 5. MERGE + VALIDATE
# =============================================================================
listings_with_rates = merge_and_validate(listings, mortgage_monthly, "Listings", LISTINGS_DATE_COL)
sold_with_rates = merge_and_validate(sold, mortgage_monthly, "Sold", SOLD_DATE_COL)


# =============================================================================
# 6. PREVIEW
# =============================================================================
header("PREVIEW")
print(listings_with_rates[[LISTINGS_DATE_COL, "year_month", "ListPrice", "rate_30yr_fixed"]].head())
print(sold_with_rates[[SOLD_DATE_COL, "year_month", "ClosePrice", "rate_30yr_fixed"]].head())


# =============================================================================
# 7. SAVE ENRICHED DATASETS
# =============================================================================
listings_output_path = os.path.join(folder_path, "listings_with_rates.csv")
sold_output_path = os.path.join(folder_path, "sold_with_rates.csv")
rates_output_path = os.path.join(folder_path, "mortgage_rates_monthly.csv")

listings_with_rates.to_csv(listings_output_path, index=False)
sold_with_rates.to_csv(sold_output_path, index=False)
mortgage_monthly.to_csv(rates_output_path, index=False)

header("DONE")
print(f"Listings saved to:      {listings_output_path}")
print(f"Sold saved to:          {sold_output_path}")
print(f"Monthly rates saved to: {rates_output_path}")