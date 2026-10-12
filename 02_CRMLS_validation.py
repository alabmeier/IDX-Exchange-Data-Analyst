import os
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:,.2f}".format)

# =============================================================================
# CONFIG
# =============================================================================
LISTINGS_FILE = "listings.csv"
SOLD_FILE = "sold.csv"

KEY_NUMERIC = [
    "ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea", "LotSizeAcres",
    "BedroomsTotal", "BathroomsTotalInteger", "DaysOnMarket", "YearBuilt",
]
LOG_SCALE = ["ClosePrice", "ListPrice", "OriginalListPrice", "LivingArea", "LotSizeAcres"]  # very right-skewed

# Core fields are kept even if mostly missing
CORE_FIELDS = set(KEY_NUMERIC) | {
    "PropertyType", "CloseDate", "ListingContractDate", "PurchaseContractDate",
    "CountyOrParish", "City", "Latitude", "Longitude", "MLSAreaMajor",
}
# Columns containing these words are treated as metadata (agents, IDs, links, ...)
METADATA_KEYWORDS = ["Agent", "Office", "Key", "Url", "Remarks", "Photo", "Mls", "Email", "Phone", "Timestamp"]

MIN_COUNTY_SALES = 30   # ignore tiny counties in the county ranking


def header(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# =============================================================================
# 1. LOAD
# =============================================================================
folder_path = input("Enter the path to the folder containing the CSV files: ")
plots_folder = os.path.join(folder_path, "eda_plots")
reports_folder = os.path.join(folder_path, "eda_reports")
os.makedirs(plots_folder, exist_ok=True)
os.makedirs(reports_folder, exist_ok=True)

data = {
    "Listings": pd.read_csv(os.path.join(folder_path, LISTINGS_FILE), low_memory=False),
    "Sold": pd.read_csv(os.path.join(folder_path, SOLD_FILE), low_memory=False),
}

# =============================================================================
# 2. DATASET UNDERSTANDING
# =============================================================================
for name, df in data.items():
    header(f"{name} - STRUCTURE")
    print(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")
    print("\nData types:")
    print(df.dtypes.value_counts().to_string())

    metadata = [c for c in df.columns if c not in CORE_FIELDS and any(k in c for k in METADATA_KEYWORDS)]
    print(f"\nMetadata fields ({len(metadata)}): {metadata}")
    print(f"Market analysis fields: {len(df.columns) - len(metadata)}")

    # Residential vs. other property types
    print("\nProperty type share (%):")
    print((df["PropertyType"].value_counts(normalize=True, dropna=False) * 100).round(2).to_string())

# =============================================================================
# 3. FILTER TO RESIDENTIAL
# =============================================================================
header("RESIDENTIAL FILTER")
for name in data:
    before = len(data[name])
    data[name] = data[name][data[name]["PropertyType"] == "Residential"].copy()
    print(f"{name}: {before:,} -> {len(data[name]):,} rows")

# =============================================================================
# 4. MISSING VALUES
# =============================================================================
for name, df in data.items():
    header(f"{name} - MISSING VALUES")
    report = pd.DataFrame({
        "Null_Count": df.isna().sum(),
        "Null_Percent": df.isna().mean() * 100,
    }).sort_values("Null_Percent", ascending=False)
    print(report.head(20))
    report.to_csv(os.path.join(reports_folder, f"{name.lower()}_missing_report.csv"))

    over_90 = report[report["Null_Percent"] > 90].index
    to_drop = [c for c in over_90 if c not in CORE_FIELDS]
    kept_core = [c for c in over_90 if c in CORE_FIELDS]
    print(f"\n>90% missing: {len(over_90)} columns")
    print(f"Dropping {len(to_drop)}: {to_drop}")
    print(f"Keeping core fields despite >90% missing: {kept_core}")

    df = df.drop(columns=to_drop)
    for col in KEY_NUMERIC:                      # make sure the key fields are numeric
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    data[name] = df

# =============================================================================
# 5. NUMERIC DISTRIBUTIONS
# =============================================================================
for name, df in data.items():
    header(f"{name} - NUMERIC DISTRIBUTIONS")
    cols = [c for c in KEY_NUMERIC if c in df.columns]

    # Percentile summary for all key fields in one table
    summary = df[cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    print(summary)
    summary.to_csv(os.path.join(reports_folder, f"{name.lower()}_distribution_summary.csv"))

    # Outliers using the IQR rule (mild = 1.5 x IQR, extreme = 3 x IQR)
    rows = {}
    for col in cols:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        rows[col] = {
            "mild_outliers": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
            "extreme_outliers": int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum()),
            "values_le_0": int((s <= 0).sum()),
        }
    outliers = pd.DataFrame(rows).T
    print(f"\n{name} - outliers:")
    print(outliers)
    outliers.to_csv(os.path.join(reports_folder, f"{name.lower()}_outlier_summary.csv"))

    # Histogram (1st-99th percentile) next to a boxplot (log scale for skewed fields)
    for col in cols:
        s = df[col].dropna()
        low, high = s.quantile([0.01, 0.99])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.hist(s[(s >= low) & (s <= high)], bins=50)
        ax1.set_title(f"{name} - {col} (1st-99th percentile)")
        if col in LOG_SCALE:
            ax2.boxplot(s[s > 0])
            ax2.set_yscale("log")
        else:
            ax2.boxplot(s)
        ax2.set_title(f"{name} - {col} boxplot")
        fig.tight_layout()
        fig.savefig(os.path.join(plots_folder, f"{name.lower()}_{col.lower()}.png"), dpi=150)
        plt.close(fig)

print(f"\nPlots saved to: {plots_folder}")

# =============================================================================
# 6. INTERN QUESTIONS (sold data)
# =============================================================================
sold = data["Sold"]
header("INTERN QUESTIONS")

# Median and average close price (ignore zero/negative prices)
prices = sold.loc[sold["ClosePrice"] > 0, "ClosePrice"]
print(f"Median close price: ${prices.median():,.0f}")
print(f"Average close price: ${prices.mean():,.0f}")

# Days on market distribution
dom = sold["DaysOnMarket"].dropna()
buckets = pd.cut(dom, bins=[-float("inf"), 0, 7, 30, 60, 90, float("inf")],
                 labels=["0 or less", "1-7", "8-30", "31-60", "61-90", "90+"])
print("\nDays on market:")
print(dom.describe().to_string())
print(buckets.value_counts().sort_index().to_string())

# Sold above vs. below list price
valid = sold[(sold["ClosePrice"] > 0) & (sold["ListPrice"] > 0)]
print(f"\nSold above list: {(valid['ClosePrice'] > valid['ListPrice']).mean() * 100:.2f}%")
print(f"Sold below list: {(valid['ClosePrice'] < valid['ListPrice']).mean() * 100:.2f}%")
print(f"Sold at list:    {(valid['ClosePrice'] == valid['ListPrice']).mean() * 100:.2f}%")

# Date consistency issues
for col in ["ListingContractDate", "PurchaseContractDate", "CloseDate"]:
    sold[col] = pd.to_datetime(sold[col], errors="coerce")
print("\nDate issues:")
print(f"Close before listing:  {(sold['CloseDate'] < sold['ListingContractDate']).sum():,}")
print(f"Purchase before listing: {(sold['PurchaseContractDate'] < sold['ListingContractDate']).sum():,}")
print(f"Close before purchase: {(sold['CloseDate'] < sold['PurchaseContractDate']).sum():,}")

# Counties with the highest median price
county = (sold[sold["ClosePrice"] > 0].groupby("CountyOrParish")["ClosePrice"]
          .agg(median_price="median", sales="count"))
county = county[county["sales"] >= MIN_COUNTY_SALES].sort_values("median_price", ascending=False)
print(f"\nTop 10 counties by median price (min {MIN_COUNTY_SALES} sales):")
print(county.head(10))

# =============================================================================
# 7. SAVE
# =============================================================================
header("DONE")
data["Listings"].to_csv(os.path.join(folder_path, "residential_listings.csv"), index=False)
data["Sold"].to_csv(os.path.join(folder_path, "residential_sold.csv"), index=False)
print("\nSaved residential_listings.csv and residential_sold.csv")