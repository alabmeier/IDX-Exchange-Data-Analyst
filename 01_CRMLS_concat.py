import glob
import os
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================
# Output files written by scripts in this project. They are excluded
# when searching for monthly files so re-running never double-counts rows.
OUTPUT_FILENAMES = {
    "listings.csv", "sold.csv",
    "residential_listings.csv", "residential_sold.csv",
    "listings_with_rates.csv", "sold_with_rates.csv",
    "listings_clean.csv", "sold_clean.csv"
}

def header(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")

def load_and_concat(files):
    """Read each file once, report per-file rows, and return (combined_df, total_rows)."""
    frames = []
    for file in files:
        df = pd.read_csv(file, low_memory=False)
        print(f"  {os.path.basename(file)}: {len(df):,} rows")
        frames.append(df)

    total_rows = sum(len(df) for df in frames)
    combined = pd.concat(frames, ignore_index=True)
    return combined, total_rows

# =============================================================================
# 1. LOCATE MONTHLY FILES
# =============================================================================
folder_path = input("Enter the path to the folder containing the monthly CSV files: ")

csv_files = sorted(
    f for f in glob.glob(os.path.join(folder_path, "*.csv"))
    if os.path.basename(f).lower() not in OUTPUT_FILENAMES
)

# Separate Listings and Sold files
listings_files = [f for f in csv_files if "listing" in os.path.basename(f).lower()]
sold_files = [f for f in csv_files if "sold" in os.path.basename(f).lower()]

header("FILES FOUND")
print(f"Number of Listings files found: {len(listings_files)}")  # 28 expected (Jan 2024 - Apr 2026)
print(f"Number of Sold files found: {len(sold_files)}")          # 28 expected (Jan 2024 - Apr 2026)

if not listings_files or not sold_files:
    raise SystemExit("Missing Listings or Sold files. Check the folder path and file names.")

# =============================================================================
# 2. READ AND CONCATENATE
# =============================================================================
header("LISTINGS - INDIVIDUAL FILES")
listings_df, total_listings_rows = load_and_concat(listings_files)

header("SOLD - INDIVIDUAL FILES")
sold_df, total_sold_rows = load_and_concat(sold_files)

# =============================================================================
# 3. VERIFY ROW COUNTS (individual files vs. concatenated)
# =============================================================================
header("ROW COUNT VERIFICATION")
print(f"Total rows from individual Listings files: {total_listings_rows:,}")
print(f"Total rows from individual Sold files: {total_sold_rows:,}")
print(f"Total rows in Listings DataFrame before Residential filter: {len(listings_df):,}")
print(f"Total rows in Sold DataFrame before Residential filter: {len(sold_df):,}")

for name, expected, actual in [
    ("Listings", total_listings_rows, len(listings_df)),
    ("Sold", total_sold_rows, len(sold_df)),
]:
    status = "OK - all rows retained" if expected == actual else "MISMATCH - rows were lost or added"
    print(f"{name}: {status}")

# =============================================================================
# 4. RESIDENTIAL FILTER
# =============================================================================
filtered_listings_df = listings_df[listings_df["PropertyType"] == "Residential"]
filtered_sold_df = sold_df[sold_df["PropertyType"] == "Residential"]

header("RESIDENTIAL FILTER")
print(f"Total rows in Listings DataFrame after Residential filter: {len(filtered_listings_df):,}")
print(f"Total rows in Sold DataFrame after Residential filter: {len(filtered_sold_df):,}")

# =============================================================================
# 5. SAVE COMBINED DATASETS
# =============================================================================
listings_output_path = os.path.join(folder_path, "listings.csv")
sold_output_path = os.path.join(folder_path, "sold.csv")

listings_df.to_csv(listings_output_path, index=False)
sold_df.to_csv(sold_output_path, index=False)

header("DONE")
print(f"Listings saved to: {listings_output_path}")
print(f"Sold saved to: {sold_output_path}")

# Previous results for reference:
# Listings: 860,898 rows total -> 547,162 Residential
# Sold:     615,707 rows total -> 414,054 Residential