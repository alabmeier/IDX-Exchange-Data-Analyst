import os
import re
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:,.2f}".format)

# =============================================================================
# CONFIG
# =============================================================================
KEY_NUMERIC = [
    "ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea",
    "LotSizeAcres", "BedroomsTotal", "BathroomsTotalInteger",
    "DaysOnMarket", "YearBuilt",
]

# Core fields are kept even if heavily missing (add/remove as needed)
CORE_FIELDS = set(KEY_NUMERIC) | {
    "PropertyType", "PropertySubType", "StandardStatus", "CloseDate",
    "ListingContractDate", "PurchaseContractDate", "CountyOrParish", "City",
    "PostalCode", "StateOrProvince", "Latitude", "Longitude",
}

# Column-name patterns that usually indicate metadata (IDs, agents, URLs, etc.)
METADATA_PATTERN = re.compile(
    r"(Key|Id(?=[A-Z]|$)|Url|URL|Agent|Office|Mls|MLS|Timestamp|Modification|"
    r"Source|Remarks|Photo|Media|Virtual|Tour|Phone|Email|Syndicat|"
    r"Disclosure|Originating|Brokerage|Member)"
)

# Heavily right-skewed fields: a linear boxplot gets squashed into a flat line at
# the bottom, so these use a log-scale y-axis (positive values only).
LOG_SCALE_FIELDS = {"ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea", "LotSizeAcres"}

# Fields required as a standalone deliverable (saved to its own CSV)
DELIVERABLE_FIELDS = ["ClosePrice", "LivingArea", "DaysOnMarket"]

MISSING_DROP_THRESHOLD = 90.0   # % missing: above this, non-core columns are dropped automatically
MISSING_REVIEW_THRESHOLD = 50.0 # % missing: between this and the drop threshold, flag for manual review (not dropped)
MIN_COUNTY_SALES = 30           # ignore tiny counties in the county ranking


def header(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# =============================================================================
# 1. LOAD DATA
# =============================================================================
folder_path = input("Enter the path to the folder containing the CSV files: ")

listings = pd.read_csv(os.path.join(folder_path, "listings.csv"), low_memory=False)
sold = pd.read_csv(os.path.join(folder_path, "sold.csv"), low_memory=False)

plots_folder = os.path.join(folder_path, "eda_plots")
reports_folder = os.path.join(folder_path, "eda_reports")
os.makedirs(plots_folder, exist_ok=True)
os.makedirs(reports_folder, exist_ok=True)

# =============================================================================
# 2. DATASET UNDERSTANDING (raw data)
# =============================================================================
for df, name in [(listings, "Listings"), (sold, "Sold")]:
    header(f"{name} - RAW STRUCTURE")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nData type counts:")
    print(df.dtypes.value_counts())

    # Q: Residential vs. other property type share
    print("\nProperty type counts:")
    print(df["PropertyType"].value_counts(dropna=False))
    print("\nProperty type share (%):")
    print((df["PropertyType"].value_counts(normalize=True, dropna=False) * 100).round(2))

# =============================================================================
# 3. FILTER TO RESIDENTIAL
# =============================================================================
residential_listings = listings[listings["PropertyType"] == "Residential"].copy()
residential_sold = sold[sold["PropertyType"] == "Residential"].copy()

header("RESIDENTIAL FILTER")
print(f"Residential Listings: {len(residential_listings):,} of {len(listings):,}")
print(f"Residential Sold:     {len(residential_sold):,} of {len(sold):,}")

datasets = [("Listings", residential_listings), ("Sold", residential_sold)]

# =============================================================================
# 4. MARKET ANALYSIS vs. METADATA FIELDS
# =============================================================================
def classify_column(col):
    if col in CORE_FIELDS:
        return "Market Analysis"
    if METADATA_PATTERN.search(col):
        return "Metadata"
    return "Market Analysis"


for name, df in datasets:
    header(f"{name} - FIELD CLASSIFICATION")
    classification = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str).values,
        "Field_Type": [classify_column(c) for c in df.columns],
    })
    print(classification["Field_Type"].value_counts())
    print("\nMetadata fields:")
    print(classification.loc[classification.Field_Type == "Metadata", "Column"].tolist())
    classification.to_csv(
        os.path.join(reports_folder, f"{name.lower()}_field_classification.csv"),
        index=False,
    )

# =============================================================================
# 5. MISSING VALUE ANALYSIS + DROP/RETAIN DECISION
# =============================================================================
cleaned = {}
for name, df in datasets:
    header(f"{name} - MISSING VALUE REPORT")

    missing = pd.DataFrame({
        "Null_Count": df.isna().sum(),
        "Null_Percent": (df.isna().mean() * 100).round(2),
    }).sort_values("Null_Percent", ascending=False)

    missing["Flag_>90%"] = missing["Null_Percent"] > MISSING_DROP_THRESHOLD
    missing["Is_Core"] = missing.index.isin(CORE_FIELDS)
    missing["Decision"] = "Retain"
    missing.loc[missing["Flag_>90%"] & ~missing["Is_Core"], "Decision"] = "Drop"
    missing.loc[missing["Flag_>90%"] & missing["Is_Core"], "Decision"] = "Retain (core)"

    # Review band: partially missing columns are kept, but listed for a manual look.
    # Blank can mean "not applicable" (e.g. pool, HOA fee), so these shouldn't be auto-dropped.
    in_review_band = (
        (missing["Null_Percent"] > MISSING_REVIEW_THRESHOLD)
        & (missing["Null_Percent"] <= MISSING_DROP_THRESHOLD)
    )
    missing.loc[in_review_band & ~missing["Is_Core"], "Decision"] = "Review"

    print(missing)
    print(f"\n{name} columns >{MISSING_DROP_THRESHOLD:.0f}% null:")
    print(missing[missing["Flag_>90%"]])

    review = missing[missing["Decision"] == "Review"]
    print(f"\n{name} columns in review band "
          f"({MISSING_REVIEW_THRESHOLD:.0f}-{MISSING_DROP_THRESHOLD:.0f}% null, kept for now): {len(review)}")
    print(review[["Null_Count", "Null_Percent"]])
    review[["Null_Count", "Null_Percent"]].to_csv(
        os.path.join(reports_folder, f"{name.lower()}_review_columns.csv")
    )

    to_drop = missing.index[missing["Decision"] == "Drop"].tolist()
    print(f"\n{name}: dropping {len(to_drop)} columns, retaining {len(df.columns) - len(to_drop)}")
    print("Dropped:", to_drop)

    missing.to_csv(os.path.join(reports_folder, f"{name.lower()}_missing_report.csv"))
    cleaned[name] = df.drop(columns=to_drop)

residential_listings = cleaned["Listings"]
residential_sold = cleaned["Sold"]
datasets = [("Listings", residential_listings), ("Sold", residential_sold)]

# =============================================================================
# 6. NUMERIC DISTRIBUTION REVIEW
# =============================================================================
PERCENTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

for name, df in datasets:
    header(f"{name} - NUMERIC DISTRIBUTIONS")
    cols = [c for c in KEY_NUMERIC if c in df.columns]
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")

    # --- Percentile summary: one compact table for all key fields ---
    summary = numeric.describe(percentiles=PERCENTILES).T
    print(summary)
    summary.to_csv(os.path.join(reports_folder, f"{name.lower()}_distribution_summary.csv"))
    if name == "Sold":
        summary.loc[[c for c in DELIVERABLE_FIELDS if c in summary.index]].to_csv(
            os.path.join(reports_folder, "sold_distribution_summary_required3.csv")
        )

    # --- Outliers (IQR rule) ---
    # IQR = Q3 - Q1. Mild fences = Q1 - 1.5*IQR / Q3 + 1.5*IQR; extreme fences use 3*IQR.
    rows = {}
    for col in cols:
        values = numeric[col].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        xlo, xhi = q1 - 3 * iqr, q3 + 3 * iqr
        rows[col] = {
            "Q1": q1, "Q3": q3, "IQR": iqr,
            "lower_fence": lo, "upper_fence": hi,
            "below_lower": int((values < lo).sum()),
            "above_upper": int((values > hi).sum()),
            "pct_outliers": round(float(((values < lo) | (values > hi)).mean() * 100), 2),
            "extreme_outliers_3xIQR": int(((values < xlo) | (values > xhi)).sum()),
            "values_le_0": int((values <= 0).sum()),
        }
    outliers = pd.DataFrame.from_dict(rows, orient="index")
    print(f"\n{name} - Outlier Summary:")
    print(outliers)
    outliers.to_csv(os.path.join(reports_folder, f"{name.lower()}_outlier_summary.csv"))

    # --- Histogram + boxplot side by side ---
    for col in cols:
        values = numeric[col].dropna()
        if values.empty:
            continue

        lo, hi = values.quantile([0.01, 0.99])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram trimmed to 1st-99th percentile so extremes don't flatten the shape
        ax1.hist(values[(values >= lo) & (values <= hi)], bins=50)
        ax1.set_title(f"{name} - {col} (1st-99th percentile)")
        ax1.set_xlabel(col)
        ax1.set_ylabel("Frequency")

        # Boxplot uses the full range so outliers stay visible
        positive = values[values > 0]
        if col in LOG_SCALE_FIELDS and not positive.empty:
            ax2.boxplot(positive)
            ax2.set_yscale("log")
            ax2.set_title(f"{name} - {col} boxplot (log scale, >0 only)")
        else:
            ax2.boxplot(values)
            ax2.set_title(f"{name} - {col} boxplot")
        ax2.set_ylabel(col)

        fig.tight_layout()
        fig.savefig(os.path.join(plots_folder, f"{name.lower()}_{col.lower()}.png"), dpi=200)
        plt.close(fig)

print(f"\nReports saved to: {reports_folder}")
print(f"Plots saved to: {plots_folder}")

# =============================================================================
# 7. SUGGESTED INTERN QUESTIONS (Sold data)
# =============================================================================
header("INTERN QUESTIONS - SOLD DATA")
s = residential_sold.copy()

# Q: Median and average close price
if "ClosePrice" in s.columns:
    # Exclude zero/negative prices: they are bad records and drag the stats down
    cp = pd.to_numeric(s["ClosePrice"], errors="coerce")
    cp = cp[cp > 0]
    print(f"\nClosePrice median: ${cp.median():,.0f}")
    print(f"ClosePrice mean:   ${cp.mean():,.0f}")
    print("(mean > median means a few very expensive homes pull the average up)")

# Q: Days on Market distribution
if "DaysOnMarket" in s.columns:
    dom = pd.to_numeric(s["DaysOnMarket"], errors="coerce").dropna()
    print("\nDaysOnMarket distribution:")
    print(dom.describe(percentiles=PERCENTILES))
    print("\nDaysOnMarket by bucket:")
    buckets = pd.cut(dom, bins=[-float("inf"), 0, 7, 14, 30, 60, 90, 180, float("inf")],
                     labels=["<=0", "1-7", "8-14", "15-30", "31-60", "61-90", "91-180", "180+"])
    print(buckets.value_counts(sort=False))

# Q: % sold above vs. below list price
if {"ClosePrice", "ListPrice"}.issubset(s.columns):
    both = s[["ClosePrice", "ListPrice"]].apply(pd.to_numeric, errors="coerce").dropna()
    both = both[(both["ListPrice"] > 0) & (both["ClosePrice"] > 0)]
    above = (both.ClosePrice > both.ListPrice).mean() * 100
    below = (both.ClosePrice < both.ListPrice).mean() * 100
    equal = (both.ClosePrice == both.ListPrice).mean() * 100
    print(f"\nSold ABOVE list: {above:.2f}%")
    print(f"Sold BELOW list: {below:.2f}%")
    print(f"Sold AT list:    {equal:.2f}%")
    print(f"(based on {len(both):,} records with both prices)")

# Q: Date consistency issues
header("DATE CONSISTENCY CHECKS - SOLD DATA")
date_cols = ["ListingContractDate", "PurchaseContractDate", "CloseDate"]
for c in date_cols:
    if c in s.columns:
        parsed = pd.to_datetime(s[c], errors="coerce")
        print(f"{c}: {s[c].isna().sum():,} missing, "
              f"{(parsed.isna() & s[c].notna()).sum():,} unparseable")
        s[c] = parsed

checks = [
    ("CloseDate", "ListingContractDate", "Close date BEFORE listing date"),
    ("PurchaseContractDate", "ListingContractDate", "Purchase contract date BEFORE listing date"),
    ("CloseDate", "PurchaseContractDate", "Close date BEFORE purchase contract date"),
]
for later, earlier, label in checks:
    if later in s.columns and earlier in s.columns:
        bad = s[s[later] < s[earlier]]
        print(f"{label}: {len(bad):,} records")
        if len(bad):
            bad[[c for c in ["ListingKey", "ListingId", earlier, later] if c in bad.columns]] \
                .to_csv(os.path.join(reports_folder, f"date_issue_{later}_before_{earlier}.csv"), index=False)

if "DaysOnMarket" in s.columns:
    print(f"Negative DaysOnMarket: {(pd.to_numeric(s['DaysOnMarket'], errors='coerce') < 0).sum():,} records")

# Q: Counties with the highest median prices
if {"CountyOrParish", "ClosePrice"}.issubset(s.columns):
    s["ClosePrice"] = pd.to_numeric(s["ClosePrice"], errors="coerce")
    county = (
        s[s["ClosePrice"] > 0].groupby("CountyOrParish")["ClosePrice"]
        .agg(Median_Price="median", Mean_Price="mean", Sales="count")
        .query("Sales >= @MIN_COUNTY_SALES")
        .sort_values("Median_Price", ascending=False)
    )
    header(f"COUNTY MEDIAN CLOSE PRICE (min {MIN_COUNTY_SALES} sales)")
    print(county.head(15).round(0))
    county.to_csv(os.path.join(reports_folder, "county_median_prices.csv"))

    top = county.head(10).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(top.index.astype(str), top["Median_Price"])
    plt.title("Top 10 Counties by Median Close Price")
    plt.xlabel("Median Close Price ($)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, "sold_top_counties_median_price.png"), dpi=300)
    plt.close()

# =============================================================================
# 8. SAVE FILTERED RESIDENTIAL DATASETS
# =============================================================================
residential_listings.to_csv(os.path.join(folder_path, "residential_listings.csv"), index=False)
residential_sold.to_csv(os.path.join(folder_path, "residential_sold.csv"), index=False)

header("DONE")
print(f"Filtered datasets saved to: {folder_path}")
print(f"Reports saved to:           {reports_folder}")
print(f"Plots saved to:             {plots_folder}")