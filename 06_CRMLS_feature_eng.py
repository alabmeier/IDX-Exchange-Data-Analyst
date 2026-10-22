import os
import glob
import pandas as pd
import geopandas as gpd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:,.2f}".format)

# =============================================================================
# CONFIG
# =============================================================================
LISTINGS_FILE = "listings_clean.csv"
SOLD_FILE = "sold_clean.csv"

DATE_COLS = ["CloseDate", "PurchaseContractDate", "ListingContractDate"]

PRICE_RATIO_DENOMINATOR = "ListPrice"

# School districts folder
SCHOOL_DISTRICT_PATH = "DistrictAreas2425"
DISTRICT_NAME_COL = "DistrictName"

SEGMENT_TABLES = [
    (["PropertyType", "PropertySubType"], "property_type"),
    (["CountyOrParish"], "county"),
    (["MLSAreaMajor"], "mls_area"),
    (["CountyOrParish", "MLSAreaMajor"], "county_mls_area"),
    (["ListOfficeName"], "list_office"),
    (["BuyerOfficeName"], "buyer_office"),
]
MIN_SEGMENT_COUNT = 30
PRINT_ROWS = 10

# The new columns (names must match what add_metrics creates)
NEW_COLS = ["PriceRatio", "Close_to_Original_List_Ratio", "PricePerSqFt", "DaysOnMarket",
            "Year", "Month", "YrMo", "Listing_to_Contract_Days", "Contract_to_Close_Days"]


def header(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}\n")


def safe_divide(numerator, denominator):
    """Divide, leaving blank when denominator is missing or <= 0."""
    return numerator / denominator.where(denominator > 0)


def add_metrics(df):
    df = df.reset_index(drop=True)

    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea", "DaysOnMarket"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Price metrics
    df["PriceRatio"] = safe_divide(df["ClosePrice"], df[PRICE_RATIO_DENOMINATOR])
    df["Close_to_Original_List_Ratio"] = safe_divide(df["ClosePrice"], df["OriginalListPrice"])
    df["PricePerSqFt"] = safe_divide(df["ClosePrice"], df["LivingArea"])

    # Time features from CloseDate
    df["Year"] = df["CloseDate"].dt.year.astype("Int64")
    df["Month"] = df["CloseDate"].dt.month.astype("Int64")
    df["YrMo"] = df["CloseDate"].dt.strftime("%Y-%m")

    # Timeline durations in days
    df["Listing_to_Contract_Days"] = (df["PurchaseContractDate"] - df["ListingContractDate"]).dt.days
    df["Contract_to_Close_Days"] = (df["CloseDate"] - df["PurchaseContractDate"]).dt.days

    # Negative durations mean the dates are out of order. Blank them out so they don't affect the averages.
    for col in ["Listing_to_Contract_Days", "Contract_to_Close_Days"]:
        negative = df[col] < 0
        df.loc[negative, col] = float("nan")
        print(f" {col}: {int(negative.sum()):,} negative values set to blank")
    return df


def find_district_file(path, data_folder):
    """Return the shapefile/GeoJSON/GeoPackage to read, given a folder or file path."""
    candidates = [path, os.path.join(data_folder, path)]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
        if os.path.isdir(candidate):
            for ext in ("*.shp", "*.geojson", "*.gpkg", "*.json"):
                found = sorted(glob.glob(os.path.join(candidate, "**", ext), recursive=True))
                if found:
                    return found[0]
            print(f"School districts: No shapefile/GeoJSON/GeoPackage found in {candidate}")
            print(f"  Contents: {os.listdir(candidate)}")
            return None

    # Only report "not found" after BOTH locations have been checked
    print(f"School districts: '{path}' not found. Looked in:")
    for candidate in candidates:
        print(f"  {os.path.abspath(candidate)}")
    return None


def load_school_districts(data_folder):
    """Load the school district boundaries from a shapefile/GeoJSON/GeoPackage."""
    file = find_district_file(SCHOOL_DISTRICT_PATH, data_folder)
    if file is None:
        return None
    print(f"Loading school districts from {file}")

    districts = gpd.read_file(file)
    print(f"Original coordinate system: {districts.crs}")
    if districts.crs is None:
        print("School districts: No coordinate system found, assuming lat/lon")
        districts = districts.set_crs(epsg=4326)
    districts = districts.to_crs(epsg=4326)

    print(f"School districts loaded: {len(districts):,} polygons")
    print(f"All columns: {list(districts.columns)}")

    # Stop with a clear message (instead of a KeyError) if the name column is wrong
    if DISTRICT_NAME_COL not in districts.columns:
        print(f"School districts: column '{DISTRICT_NAME_COL}' not found. "
              f"Set DISTRICT_NAME_COL to one of the columns listed above.")
        return None
    print(f"Using name column '{DISTRICT_NAME_COL}'")
    return districts[[DISTRICT_NAME_COL, "geometry"]].rename(columns={DISTRICT_NAME_COL: "SchoolDistrict"})


def add_school_districts(df, districts):
    """Add a SchoolDistrict column to the DataFrame based on the property coordinates."""

    # Skip rows flagged with bad coordinates in cleaning step
    valid = df["Latitude"].notna() & df["Longitude"].notna()
    if "geo_any_invalid_flag" in df.columns:
        valid &= ~df["geo_any_invalid_flag"].astype(bool)

    points = gpd.GeoDataFrame(
        df.loc[valid, []],
        geometry=gpd.points_from_xy(df.loc[valid, "Longitude"], df.loc[valid, "Latitude"]),
        crs="EPSG:4326"
    )
    joined = gpd.sjoin(points, districts, how="left", predicate="within")

    # Elementary and high school districts overlap, so a home can fall in more than one.
    # Join all matching names into one text value to keep one row per property.
    names = joined.groupby(level=0)["SchoolDistrict"].agg(
        lambda s: " | ".join(sorted(set(s.dropna()))) or None
    )
    df["SchoolDistrict"] = names.reindex(df.index)

    # regex=False matters: "|" on its own is a regex that matches every row
    multiple = df["SchoolDistrict"].str.contains(" | ", regex=False, na=False).sum()
    print(f"School district matched: {df['SchoolDistrict'].notna().sum():,} | "
          f"no match: {df.loc[valid, 'SchoolDistrict'].isna().sum():,} | "
          f"bad/missing coordinates: {len(df) - valid.sum():,} | "
          f"in more than one district: {multiple:,}")
    return df


def segment_summary(df, by):
    """Count, total volume, and typical metrics for every combination of the 'by' columns."""
    wanted = {
        "TotalVolume": ("ClosePrice", "sum"),
        "MedianClosePrice": ("ClosePrice", "median"),
        "MedianListPrice": ("ListPrice", "median"),
        "MedianPricePerSqFt": ("PricePerSqFt", "median"),
        "MedianDOM": ("DaysOnMarket", "median"),
        "AvgPriceRatio": ("PriceRatio", "mean"),
        "AvgClosetoOriginalListRatio": ("Close_to_Original_List_Ratio", "mean"),
        "MedianListingtoContractDays": ("Listing_to_Contract_Days", "median"),
        "MedianContracttoCloseDays": ("Contract_to_Close_Days", "median"),
    }
    # Skip metrics that are completely empty
    stats = {name: spec for name, spec in wanted.items() if df[spec[0]].notna().any()}

    grouped = df.groupby(by, dropna=False)
    summary = grouped.agg(**stats)
    summary.insert(0, "count", grouped.size())
    summary = summary[summary["count"] >= MIN_SEGMENT_COUNT].sort_values("count", ascending=False)
    return summary.round(3)


# =============================================================================
# 1. RUN
# =============================================================================
folder_path = input("Enter the path to the folder containing the CSV files: ")
reports_folder = os.path.join(folder_path, "feature_reports")
os.makedirs(reports_folder, exist_ok=True)

data = {
    "Listings": pd.read_csv(os.path.join(folder_path, LISTINGS_FILE), low_memory=False),
    "Sold": pd.read_csv(os.path.join(folder_path, SOLD_FILE), low_memory=False),
}

header("SCHOOL DISTRICT DATA")
districts = load_school_districts(folder_path)

for name, df in data.items():
    # 1. Engineer the metrics
    header(f"{name} - FEATURE ENGINEERING")
    df = add_metrics(df)
    if districts is not None:
        df = add_school_districts(df, districts)
    data[name] = df

    # 2. Check new columns
    print("\nNew column coverage and ranges:")
    numeric_new = [c for c in NEW_COLS if c not in ("YrMo", "Year", "Month")]
    check = df[numeric_new].describe().T[["count", "min", "50%", "max"]]
    check["missing"] = len(df) - check["count"]
    print(check)
    print(f"Year/Month/YrMo missing: {df['YrMo'].isna().sum():,} | "
          f"years: {sorted(df['Year'].dropna().unique().tolist())}")

    # 3. Sample output table with new columns
    show = [c for c in ["ListingKey", "ClosePrice", "OriginalListPrice", "LivingArea", "CloseDate"] if c in df.columns]
    show += NEW_COLS + (["SchoolDistrict"] if "SchoolDistrict" in df.columns else [])
    populated = df.dropna(subset=NEW_COLS)
    sample = (populated if len(populated) else df)[show].head(PRINT_ROWS)
    print(f"\nSample of first {PRINT_ROWS} rows with new columns:")
    print(sample.to_string(index=False))
    sample.to_csv(os.path.join(reports_folder, f"{name.lower()}_sample_output.csv"), index=False)

    # 4. Segment analysis
    header(f"{name} - SEGMENT ANALYSIS")
    for by, label in SEGMENT_TABLES:
        if not all(col in df.columns for col in by):
            print(f"Skipping segment '{label}' because one or more columns are missing: {by}")
            continue
        table = segment_summary(df, by)
        table.to_csv(os.path.join(reports_folder, f"{name.lower()}_segment_{label}.csv"))
        print(f"\nBy {' + '.join(by)} (length = {len(table):,} segments with {MIN_SEGMENT_COUNT:,}+ rows, "
              f"top {PRINT_ROWS:,} shown, full table saved):")
        print(table.head(PRINT_ROWS).to_string() if len(table) else "No segments large enough")


# =============================================================================
# 2. SAVE DATA
# =============================================================================
data["Listings"].to_csv(os.path.join(folder_path, "listings_features.csv"), index=False)
data["Sold"].to_csv(os.path.join(folder_path, "sold_features.csv"), index=False)
print(f"\nSaved listings_features.csv and sold_features.csv with new metrics to {folder_path}")