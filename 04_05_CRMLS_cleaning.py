import os
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================
LISTINGS_FILE = "listings_with_rates.csv"
SOLD_FILE = "sold_with_rates.csv"


DATE_COLS = ["CloseDate", "PurchaseContractDate", "ListingContractDate", "ContractStatusChangeDate"]
NUMERIC_COLS = [
    "ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea", "LotSizeAcres",
    "BedroomsTotal", "BathroomsTotalInteger", "DaysOnMarket", "YearBuilt",
    "Latitude", "Longitude",
]

# Rows missing any of these are dropped
REQUIRED = {
    "Sold": ["ClosePrice", "CloseDate"],
    "Listings": ["ListPrice", "ListingContractDate"],
}
TEXT_FILL_COLS = ["CountyOrParish", "City", "PostalCode"]   # blanks become "Unknown"
DROP_COLS = []                                              # columns to drop by hand

# California bounding box for the "out of state" check
CA_LAT = (32.5, 42.0)
CA_LON = (-124.5, -114.1)


def header(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def clean(df, name, folder):
    header(f"{name} - CLEANING")
    df = df.copy()
    print(f"Start: {len(df):,} rows, {df.shape[1]} columns")

    # 1. Merge duplicate-header columns (CloseDate.1 -> CloseDate)
    #    Some monthly files repeat a header, and pandas renames the copy with ".1".
    #    Fill gaps in the original from the copy, then drop the copy.
    for col in [c for c in df.columns if c.endswith(".1")]:
        base = col[:-2]
        if base in df.columns:
            df[base] = df[base].fillna(df[col])
            df = df.drop(columns=col)
            print(f"Merged {col} into {base}")

    # 2. Keep Residential only, remove exact duplicate rows
    df = df[df["PropertyType"] == "Residential"]
    before = len(df)
    df = df.drop_duplicates()
    print(f"Residential rows: {before:,} | exact duplicates removed: {before - len(df):,}")

    # 3. Convert date fields to datetime (bad values become blank)
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 4. Convert numeric fields to numbers (bad values become blank)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Drop columns that are more than 90% empty (never the date/numeric fields above)
    missing_pct = df.isna().mean() * 100
    protected = set(DATE_COLS + NUMERIC_COLS + ["PropertyType"])
    to_drop = [c for c in df.columns if missing_pct[c] > 90 and c not in protected]
    to_drop += [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"Dropped {len(to_drop)} columns: {to_drop}")

    # 6. Missing values
    #    Rows without a price or date can't be analyzed, so drop them.
    #    Numeric blanks are left alone (filling them would invent data).
    required = [c for c in REQUIRED[name] if c in df.columns]
    before = len(df)
    df = df.dropna(subset=required)
    print(f"Rows missing {required} removed: {before - len(df):,}")
    for col in TEXT_FILL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # 7. Remove invalid numeric values
    rules = {
        "ClosePrice": lambda s: s <= 0,
        "LivingArea": lambda s: s <= 0,
        "DaysOnMarket": lambda s: s < 0,
        "BedroomsTotal": lambda s: s < 0,
        "BathroomsTotalInteger": lambda s: s < 0,
    }
    invalid = pd.Series(False, index=df.index)
    for col, rule in rules.items():
        if col in df.columns:
            bad = rule(df[col])
            print(f"Invalid {col}: {int(bad.sum()):,}")
            invalid |= bad
    df.loc[invalid].to_csv(os.path.join(folder, f"{name.lower()}_removed_invalid_rows.csv"), index=False)
    df = df[~invalid]
    print(f"Rows removed as invalid: {int(invalid.sum()):,}")

    # 8. Date consistency flags (Listing <= Purchase <= Close)
    if all(c in df.columns for c in ["ListingContractDate", "PurchaseContractDate", "CloseDate"]):
        listing_after_purchase = df["ListingContractDate"] > df["PurchaseContractDate"]
        df["listing_after_close_flag"] = df["ListingContractDate"] > df["CloseDate"]
        df["purchase_after_close_flag"] = df["PurchaseContractDate"] > df["CloseDate"]
        df["negative_timeline_flag"] = (
            df["listing_after_close_flag"] | df["purchase_after_close_flag"] | listing_after_purchase
        )
    else:
        print("A date column is missing: date flags skipped")

    # 9. Geographic flags (rows are kept, only marked)
    lat, lon = df["Latitude"], df["Longitude"]
    df["geo_missing_coords_flag"] = lat.isna() | lon.isna()
    df["geo_zero_coord_flag"] = (lat == 0) | (lon == 0)
    df["geo_positive_longitude_flag"] = lon > 0
    has_real_coords = lat.notna() & lon.notna() & ~df["geo_zero_coord_flag"]
    outside_ca = ~lat.between(*CA_LAT) | ~lon.between(*CA_LON)
    df["geo_out_of_state_flag"] = has_real_coords & outside_ca

    flags = [c for c in df.columns if c.endswith("_flag")]
    geo_flags = [c for c in flags if c.startswith("geo_")]
    df["geo_any_invalid_flag"] = df[geo_flags].any(axis=1)

    # Summaries
    print("\nFlag counts:")
    print(df[flags + ["geo_any_invalid_flag"]].sum().to_string())

    id_cols = [c for c in ["ListingKey", "City", "CountyOrParish"] if c in df.columns]
    bad_coords = df.loc[df["geo_any_invalid_flag"], id_cols + ["Latitude", "Longitude"] + geo_flags]
    bad_coords.to_csv(os.path.join(folder, f"{name.lower()}_invalid_coordinates.csv"), index=False)
    print(f"Invalid coordinate records saved: {len(bad_coords):,}")

    print("\nData type confirmation:")
    check_cols = [c for c in DATE_COLS + NUMERIC_COLS if c in df.columns]
    print(df[check_cols].dtypes.to_string())

    print(f"\nEnd: {len(df):,} rows, {df.shape[1]} columns")
    return df


# =============================================================================
# 1. RUN SCRIPT
# =============================================================================
folder_path = input("Enter the path to the folder containing the CSV files: ")
reports_folder = os.path.join(folder_path, "cleaning_reports")
os.makedirs(reports_folder, exist_ok=True)

listings = pd.read_csv(os.path.join(folder_path, LISTINGS_FILE), low_memory=False)
sold = pd.read_csv(os.path.join(folder_path, SOLD_FILE), low_memory=False)

listings_clean = clean(listings, "Listings", reports_folder)
sold_clean = clean(sold, "Sold", reports_folder)


listings_clean.to_csv(os.path.join(folder_path, "listings_clean.csv"), index=False)
sold_clean.to_csv(os.path.join(folder_path, "sold_clean.csv"), index=False)

header("BEFORE / AFTER ROW COUNTS")
print(f"Listings: {len(listings):,} -> {len(listings_clean):,}")
print(f"Sold:     {len(sold):,} -> {len(sold_clean):,}")
header("DONE")
