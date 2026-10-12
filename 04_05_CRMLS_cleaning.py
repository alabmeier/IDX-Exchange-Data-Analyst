"""
Every transformation, and why it is done:
  0. Merge suffix-duplicate columns (e.g. CloseDate.1)  -> repeated headers in the monthly files
                                                           would otherwise leave values stranded.

  1. Residential filter + exact duplicate row removal    -> only residential records are analyzed;
                                                           duplicates would double count sales.

  2. Date fields -> datetime                             -> enables date math and ordering checks.

  3. Numeric fields -> numeric dtype                     -> text in numeric columns breaks stats.

  4. Drop unnecessary/redundant columns                  -> >90% missing, constant, or duplicate
                                                           columns carry no information.

  5. Handle missing values                               -> drop rows missing price/date, fill blank
                                                           text with "Unknown", never impute numerics.

  6. Invalid numeric values (removed or flagged)         -> prices/areas <= 0, negative days or
                                                           bed/bath counts are data errors.

  7. Date consistency flags                              -> listing <= purchase <= close must hold.

  8. Geographic flags                                    -> missing, zero, positive-longitude and
                                                           out-of-California coordinates are errors.
                                                           
Everything printed is also saved to cleaning_reports/cleaning_report.txt.
"""
import operator
import os
import re
import sys
import pandas as pd

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:,.2f}".format)

# =============================================================================
# CONFIG
# =============================================================================
# Input files (with rates)
LISTINGS_FILE = "listings_with_rates.csv"
SOLD_FILE = "sold_with_rates.csv"


RESIDENTIAL_ONLY = True

DATE_COLS = ["CloseDate", "PurchaseContractDate", "ListingContractDate", "ContractStatusChangeDate"]

KEY_NUMERIC = [
    "ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea",
    "LotSizeAcres", "BedroomsTotal", "BathroomsTotalInteger",
    "DaysOnMarket", "YearBuilt",
]

EXTRA_NUMERIC = [
    "Latitude", "Longitude", "LotSizeSquareFeet", "LotSizeArea", "GarageSpaces",
    "ParkingTotal", "Stories", "BathroomsFull", "BathroomsHalf", "AssociationFee",
    "TaxAnnualAmount", "rate_30yr_fixed",
]

# Core fields are never dropped as unnecessary, even if mostly missing
CORE_FIELDS = set(KEY_NUMERIC) | set(DATE_COLS) | {
    "PropertyType", "PropertySubType", "StandardStatus", "CountyOrParish", "City",
    "PostalCode", "StateOrProvince", "ListingKey", "ListingId", "year_month", "rate_30yr_fixed",
    "MLSAreaMajor", "MlsStatus"
}

# --- Column removal ---
MISSING_DROP_THRESHOLD = 90.0   # % missing: drop non-core columns above this
METADATA_PATTERN = re.compile(
    r"(Key|Id(?=[A-Z]|$)|Url|URL|Agent|Office|Mls|MLS|Timestamp|Modification|"
    r"Source|Remarks|Photo|Media|Virtual|Tour|Phone|Email|Syndicat|"
    r"Disclosure|Originating|Brokerage|Member)"
)
DROP_METADATA_COLUMNS = False
EXTRA_DROP_COLUMNS = []

# --- Missing values ---
REQUIRED_FIELDS = {
    "Sold": ["ClosePrice", "CloseDate"],
    "Listings": ["ListPrice", "ListingContractDate"],
}
# Text columns where a blank is filled with "Unknown"
FILL_UNKNOWN_COLS = ["CountyOrParish", "City", "PostalCode", "PropertySubType", "StandardStatus"]

# --- Invalid numeric values (column, operator, threshold): value is INVALID if it matches ---
INVALID_RULES = [
    ("ClosePrice", "<=", 0),
    ("LivingArea", "<=", 0),
    ("DaysOnMarket", "<", 0),
    ("BedroomsTotal", "<", 0),
    ("BathroomsTotalInteger", "<", 0),
]
INVALID_ACTION = "remove"

# --- Geographic plausibility: California bounding box (approximate) ---
CA_LAT_RANGE = (32.5, 42.0)
CA_LON_RANGE = (-124.5, -114.1)

OPS = {"<=": operator.le, "<": operator.lt, ">=": operator.ge, ">": operator.gt}
SUFFIX_DUP = re.compile(r"^(?P<base>.+)\.\d+$")

GEO_FLAGS = [
    "geo_missing_coords_flag", "geo_zero_coord_flag",
    "geo_positive_longitude_flag", "geo_out_of_state_flag",
]
DATE_FLAGS = ["listing_after_close_flag", "purchase_after_close_flag", "negative_timeline_flag"]


# =============================================================================
# HELPERS
# =============================================================================
class Tee:
    """Send everything printed to both the screen and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def header(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def merge_suffix_duplicates(df, name, reports_folder):
    """
    Columns like 'CloseDate.1' appear when a source file has two columns with the same
    header (pandas renames the second one). Fill any gaps in the base column from the
    suffixed copy, then drop the copy. Where both have a value and they disagree,
    the base column wins.
    """
    df = df.copy()
    rows = []
    for col in list(df.columns):
        m = SUFFIX_DUP.match(col)
        if not m or m.group("base") not in df.columns:
            continue
        base = m.group("base")
        both = df[base].notna() & df[col].notna()
        conflicts = int((df.loc[both, base].astype(str) != df.loc[both, col].astype(str)).sum())
        filled = int((df[base].isna() & df[col].notna()).sum())
        df[base] = df[base].fillna(df[col])
        df = df.drop(columns=col)
        rows.append({"Duplicate": col, "Merged_Into": base,
                     "Both_Filled": int(both.sum()), "Conflicts": conflicts, "Gaps_Filled": filled})

    report = pd.DataFrame(rows)
    print(f"\nSuffix-duplicate columns merged: {len(report)}")
    if len(report):
        print(report.to_string(index=False))
        report.to_csv(os.path.join(reports_folder, f"{name.lower()}_suffix_duplicates.csv"), index=False)
    return df


def find_duplicate_columns(df):
    """Return {duplicate_column: column_it_copies} for columns with identical contents."""
    groups = {}
    for col in df.columns:
        try:
            h = int(pd.util.hash_pandas_object(df[col], index=False).sum())
        except TypeError:
            continue
        groups.setdefault((h, str(df[col].dtype)), []).append(col)

    dupes = {}
    for cols in groups.values():
        keep = cols[0]
        for other in cols[1:]:
            if df[keep].equals(df[other]):
                dupes[other] = keep
    return dupes


def add_date_flags(df):
    """Listing <= Purchase <= Close must hold. Missing dates never trigger a flag."""
    def later_than(earlier, later):
        if earlier in df.columns and later in df.columns:
            return (df[earlier] > df[later]).fillna(False)
        return pd.Series(False, index=df.index)

    df["listing_after_close_flag"] = later_than("ListingContractDate", "CloseDate")
    df["purchase_after_close_flag"] = later_than("PurchaseContractDate", "CloseDate")
    listing_after_purchase = later_than("ListingContractDate", "PurchaseContractDate")
    # Negative timeline = ANY violation of Listing <= Purchase <= Close
    df["negative_timeline_flag"] = (
        df["listing_after_close_flag"] | df["purchase_after_close_flag"] | listing_after_purchase
    )
    return df


def add_geographic_flags(df):
    """Flag (never remove) missing, zero, positive-longitude and out-of-California coordinates."""
    lat, lon = df["Latitude"], df["Longitude"]
    df["geo_missing_coords_flag"] = lat.isna() | lon.isna()
    df["geo_zero_coord_flag"] = (lat == 0) | (lon == 0)
    df["geo_positive_longitude_flag"] = lon > 0

    # Out of state: real-looking coordinates (present, non-zero) outside the California box.
    checkable = lat.notna() & lon.notna() & ~df["geo_zero_coord_flag"]
    outside = ~lat.between(*CA_LAT_RANGE) | ~lon.between(*CA_LON_RANGE)
    df["geo_out_of_state_flag"] = checkable & outside

    df["geo_any_invalid_flag"] = df[GEO_FLAGS].any(axis=1)
    return df


def clean_dataset(df, name, reports_folder):
    log = []

    def record(step, why):
        prev_rows = log[-1]["Rows"] if log else len(df)
        log.append({"Step": step, "Why": why, "Rows": len(df),
                    "Rows_Removed": prev_rows - len(df), "Columns": len(df.columns)})

    header(f"{name} - CLEANING")
    record("Loaded", "starting point")

    # ---- 0. Merge columns duplicated by repeated headers (e.g. CloseDate.1) ----
    df = merge_suffix_duplicates(df, name, reports_folder)
    record("Merged suffix-duplicate columns", "repeated headers in source files")

    # ---- 1. Residential filter and exact duplicate rows ----
    if RESIDENTIAL_ONLY and "PropertyType" in df.columns:
        df = df[df["PropertyType"] == "Residential"].copy()
        record("Residential filter", "analysis covers residential properties only")

    dup_rows = int(df.duplicated().sum())
    print(f"\nExact duplicate rows removed: {dup_rows:,}")
    df = df.drop_duplicates().copy()
    record("Removed duplicate rows", "duplicates would double count transactions")

    if "ListingKey" in df.columns:
        key_dupes = int(df["ListingKey"].duplicated().sum())
        print(f"Repeated ListingKey values (kept, review manually): {key_dupes:,}")

    # ---- 2. Convert date fields to datetime ----
    print("\nDate conversion:")
    for col in DATE_COLS:
        if col not in df.columns:
            continue
        non_null_before = df[col].notna().sum()
        try:
            df[col] = pd.to_datetime(df[col], errors="coerce", format="mixed")
        except (TypeError, ValueError):  # older pandas without format="mixed"
            df[col] = pd.to_datetime(df[col], errors="coerce")
        failed = int(non_null_before - df[col].notna().sum())
        print(f"  {col}: {df[col].isna().sum():,} missing ({failed:,} unparseable)")
    record("Converted dates to datetime", "enables date math and ordering checks")

    # ---- 3. Ensure numeric fields are properly typed ----
    print("\nNumeric conversion:")
    for col in [c for c in KEY_NUMERIC + EXTRA_NUMERIC if c in df.columns]:
        non_null_before = df[col].notna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        failed = int(non_null_before - df[col].notna().sum())
        print(f"  {col}: dtype={df[col].dtype}, {failed:,} non-numeric values set to NaN")
    record("Converted numerics", "text in numeric columns breaks statistics")

    # ---- 4. Remove unnecessary or redundant columns ----
    drop_reasons = {}

    missing_pct = df.isna().mean() * 100
    for col in missing_pct.index[missing_pct > MISSING_DROP_THRESHOLD]:
        if col not in CORE_FIELDS:
            drop_reasons[col] = f">{MISSING_DROP_THRESHOLD:.0f}% missing"

    for col in df.columns:
        if col not in CORE_FIELDS and col not in drop_reasons and df[col].nunique(dropna=False) <= 1:
            drop_reasons[col] = "constant (one value)"

    remaining = df.drop(columns=list(drop_reasons))
    for dup, original in find_duplicate_columns(remaining).items():
        if dup not in CORE_FIELDS:
            drop_reasons[dup] = f"duplicate of {original}"

    metadata_cols = [c for c in df.columns
                     if c not in CORE_FIELDS and c not in drop_reasons and METADATA_PATTERN.search(c)]
    if DROP_METADATA_COLUMNS:
        for col in metadata_cols:
            drop_reasons.setdefault(col, "metadata")
    else:
        print(f"\nMetadata-looking columns kept (set DROP_METADATA_COLUMNS=True to drop): {len(metadata_cols)}")

    for col in EXTRA_DROP_COLUMNS:
        if col in df.columns:
            drop_reasons.setdefault(col, "manual drop")

    drop_report = pd.DataFrame({"Column": list(drop_reasons), "Reason": list(drop_reasons.values())})
    print(f"\nDropping {len(drop_report)} columns:")
    print(drop_report.to_string(index=False) if len(drop_report) else "  none")
    drop_report.to_csv(os.path.join(reports_folder, f"{name.lower()}_dropped_columns.csv"), index=False)

    df = df.drop(columns=list(drop_reasons))
    record("Dropped unnecessary/redundant columns", "no information (empty, constant, or duplicate)")

    # ---- 5. Handle missing values ----
    required = [c for c in REQUIRED_FIELDS.get(name, []) if c in df.columns]
    missing_required = df[required].isna().any(axis=1)
    print(f"\nRows missing a required field {required}: {int(missing_required.sum()):,} (removed)")
    df = df[~missing_required].copy()
    record("Removed rows missing required fields", "a record without price/date can't be analyzed")

    filled = {}
    for col in FILL_UNKNOWN_COLS:
        if col in df.columns and df[col].isna().any():
            filled[col] = int(df[col].isna().sum())
            df[col] = df[col].fillna("Unknown")
    print(f"Blank text values filled with 'Unknown': {filled if filled else 'none'}")

    # Numeric fields are not imputed; rows are flagged instead so you can decide later
    numeric_present = [c for c in KEY_NUMERIC if c in df.columns]
    df["missing_key_numeric_count"] = df[numeric_present].isna().sum(axis=1)
    print(f"Rows with at least one missing key numeric field (kept, flagged): "
          f"{int((df['missing_key_numeric_count'] > 0).sum()):,}")
    record("Handled missing values", "numerics not imputed (would invent data); text filled with Unknown")

    # ---- 6. Remove or flag invalid numeric values ----
    flag_cols = []
    print("\nInvalid numeric values:")
    for col, op, threshold in INVALID_RULES:
        if col not in df.columns:
            continue
        flag = f"invalid_{col}"
        df[flag] = OPS[op](df[col], threshold).fillna(False)
        flag_cols.append(flag)
        print(f"  {col} {op} {threshold}: {int(df[flag].sum()):,} rows")

    if flag_cols:
        df["has_invalid_numeric"] = df[flag_cols].any(axis=1)
        n_invalid = int(df["has_invalid_numeric"].sum())
        print(f"  Rows with any invalid value: {n_invalid:,}")

        if INVALID_ACTION == "remove":
            df[df["has_invalid_numeric"]].to_csv(
                os.path.join(reports_folder, f"{name.lower()}_removed_invalid_rows.csv"), index=False
            )
            df = df[~df["has_invalid_numeric"]].drop(columns=flag_cols + ["has_invalid_numeric"]).copy()
            print("  Action: removed (removed rows saved to reports folder)")
        else:
            print("  Action: flagged (rows kept; filter on has_invalid_numeric later)")
    record(f"Invalid numerics ({INVALID_ACTION})", "price/area <= 0, negative days or bed/bath are errors")

    # ---- 7. Date consistency flags ----
    df = add_date_flags(df)
    print("\nDate consistency flag counts (rows kept):")
    date_summary = pd.DataFrame({
        "Flag": DATE_FLAGS,
        "Count": [int(df[f].sum()) for f in DATE_FLAGS],
    })
    date_summary["Percent"] = date_summary["Count"] / max(len(df), 1) * 100
    print(date_summary.to_string(index=False))
    date_summary.to_csv(os.path.join(reports_folder, f"{name.lower()}_date_flag_summary.csv"), index=False)
    record("Date consistency flags", "listing <= purchase <= close must hold")

    # ---- 8. Geographic data checks ----
    if {"Latitude", "Longitude"}.issubset(df.columns):
        df = add_geographic_flags(df)
        geo_rows = [{"Check": f, "Count": int(df[f].sum())} for f in GEO_FLAGS + ["geo_any_invalid_flag"]]
        geo_rows.append({"Check": "valid_coordinates", "Count": int((~df["geo_any_invalid_flag"]).sum())})
        geo_summary = pd.DataFrame(geo_rows)
        geo_summary["Percent"] = geo_summary["Count"] / max(len(df), 1) * 100
        print(f"\nGeographic data quality summary (California box: lat {CA_LAT_RANGE}, lon {CA_LON_RANGE}):")
        print(geo_summary.to_string(index=False))
        geo_summary.to_csv(os.path.join(reports_folder, f"{name.lower()}_geo_quality_summary.csv"), index=False)

        id_cols = [c for c in ["ListingKey", "ListingId", "City", "CountyOrParish", "PostalCode"] if c in df.columns]
        invalid_records = df.loc[df["geo_any_invalid_flag"], id_cols + ["Latitude", "Longitude"] + GEO_FLAGS]
        invalid_records.to_csv(os.path.join(reports_folder, f"{name.lower()}_invalid_coordinate_records.csv"),
                               index=False)
        print(f"Invalid coordinate records saved: {len(invalid_records):,}")
        if len(invalid_records):
            print(invalid_records.head(10).to_string(index=False))
    else:
        print("\nLatitude/Longitude not found: geographic checks skipped")
    record("Geographic flags", "missing/zero/positive/out-of-state coordinates are errors")

    # ---- Data type confirmation ----
    expected = [(c, "datetime") for c in DATE_COLS if c in df.columns]
    expected += [(c, "numeric") for c in KEY_NUMERIC + EXTRA_NUMERIC if c in df.columns]
    rows = []
    for col, kind in expected:
        ok = (pd.api.types.is_datetime64_any_dtype(df[col]) if kind == "datetime"
              else pd.api.types.is_numeric_dtype(df[col]))
        rows.append({"Column": col, "Expected": kind, "Actual": str(df[col].dtype),
                     "Status": "PASS" if ok else "FAIL"})
    dtype_check = pd.DataFrame(rows)
    print("\nData type confirmation:")
    print(dtype_check.to_string(index=False))
    print(f"Result: {int((dtype_check['Status'] == 'PASS').sum())} PASS, "
          f"{int((dtype_check['Status'] == 'FAIL').sum())} FAIL")
    dtype_check.to_csv(os.path.join(reports_folder, f"{name.lower()}_dtype_confirmation.csv"), index=False)

    # ---- Transformation log ----
    summary = pd.DataFrame(log)
    print("\nTransformation log (rows/columns after each step):")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(reports_folder, f"{name.lower()}_transformation_log.csv"), index=False)

    missing_after = pd.DataFrame({
        "Null_Count": df.isna().sum(),
        "Null_Percent": df.isna().mean() * 100,
    }).sort_values("Null_Percent", ascending=False)
    missing_after.to_csv(os.path.join(reports_folder, f"{name.lower()}_missing_after_cleaning.csv"))

    return df


# =============================================================================
# 1. LOAD DATA
# =============================================================================
folder_path = input("Enter the path to the folder containing the CSV files: ")
reports_folder = os.path.join(folder_path, "cleaning_reports")
os.makedirs(reports_folder, exist_ok=True)

# Save everything printed from here on to cleaning_report.txt as well
log_file = open(os.path.join(reports_folder, "cleaning_report.txt"), "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, log_file)

listings = pd.read_csv(os.path.join(folder_path, LISTINGS_FILE), low_memory=False)
sold = pd.read_csv(os.path.join(folder_path, SOLD_FILE), low_memory=False)


header("DATA LOADED")
print(f"Listings: {listings.shape[0]:,} rows x {listings.shape[1]:,} columns")
print(f"Sold:     {sold.shape[0]:,} rows x {sold.shape[1]:,} columns")


# =============================================================================
# 2. CLEAN
# =============================================================================
listings_clean = clean_dataset(listings, "Listings", reports_folder)
sold_clean = clean_dataset(sold, "Sold", reports_folder)

# =============================================================================
# 3. SAVE
# =============================================================================
listings_output_path = os.path.join(folder_path, "listings_clean.csv")
sold_output_path = os.path.join(folder_path, "sold_clean.csv")

listings_clean.to_csv(listings_output_path, index=False)
sold_clean.to_csv(sold_output_path, index=False)

# =============================================================================
# 4. BEFORE / AFTER SUMMARY
# =============================================================================
header("BEFORE / AFTER SUMMARY")
ba = pd.DataFrame([
    {"Dataset": "Listings", "Rows_Before": len(listings), "Rows_After": len(listings_clean),
     "Cols_Before": listings.shape[1], "Cols_After": listings_clean.shape[1]},
    {"Dataset": "Sold", "Rows_Before": len(sold), "Rows_After": len(sold_clean),
     "Cols_Before": sold.shape[1], "Cols_After": sold_clean.shape[1]}
    
])
ba["Rows_Removed"] = ba["Rows_Before"] - ba["Rows_After"]
print(ba.to_string(index=False))
ba.to_csv(os.path.join(reports_folder, "before_after_summary.csv"), index=False)

print(f"\nListings saved to: {listings_output_path}")
print(f"Sold saved to:     {sold_output_path}")
print(f"Reports saved to:  {reports_folder}")

sys.stdout = sys.__stdout__
log_file.close()