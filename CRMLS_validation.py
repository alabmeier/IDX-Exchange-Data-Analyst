import pandas as pd
import matplotlib.pyplot as plt
import os


# Load data
folder_path = input("Enter the path to the folder containing the CSV files: ")

listings = pd.read_csv(
    os.path.join(folder_path, "listings.csv"),
    low_memory=False
)

sold = pd.read_csv(
    os.path.join(folder_path, "sold.csv"),
    low_memory=False
)


# Basic information
for df, name in [(listings, "Listings"), (sold, "Sold")]:
    print(f"\n{'=' * 50}")
    print(name)
    print("=" * 50)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nProperty types:")
    print(df["PropertyType"].value_counts())


# Residential data
residential_listings = listings[
    listings["PropertyType"] == "Residential"
]

residential_sold = sold[
    sold["PropertyType"] == "Residential"
]

print("\nResidential Listings:", f"{len(residential_listings):,}")
print("Residential Sold:", f"{len(residential_sold):,}")

# Missing value analysis
for df, name in [
    (residential_listings, "Listings"),
    (residential_sold, "Sold")
]:

    missing = pd.DataFrame({
        "Null_Count": df.isna().sum(),
        "Null_Percent": df.isna().mean() * 100
    }).sort_values("Null_Percent", ascending=False)

    print(f"\n{name} Missing Value Report:")
    print(missing)

    print(f"\n{name} Columns >90% Null:")
    print(missing[missing["Null_Percent"] > 90])


# Numeric distributions
key_numeric_columns = [
    "ClosePrice",
    "ListPrice",
    "OriginalListPrice",
    "LivingArea",
    "LotSizeAcres",
    "BedroomsTotal",
    "BathroomsTotalInteger",
    "DaysOnMarket",
    "YearBuilt"
]

# Create folder for EDA plots
plots_folder = os.path.join(folder_path, "eda_plots")
os.makedirs(plots_folder, exist_ok=True)

for df, name in [
    (residential_listings, "Listings"),
    (residential_sold, "Sold")
]:

    columns = [
        col for col in key_numeric_columns
        if col in df.columns
    ]

    for col in columns:

        values = df[col].dropna()

        # Numeric summary
        print(f"\n{name} - {col} Summary:")
        print(
            values.describe(
                percentiles=[0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            )
        )

        # Histogram
        plt.figure(figsize=(8, 5))

        lower = values.quantile(0.01)
        upper = values.quantile(0.99)

        plt.hist(
            values[(values >= lower) & (values <= upper)],
            bins=50
        )

        plt.title(f"{name} - {col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                plots_folder,
                f"{name.lower()}_{col.lower()}_histogram.png"
            ),
            dpi=300
        )

        plt.close()

        # Boxplot
        plt.figure(figsize=(8, 5))
        plt.boxplot(values)
        plt.title(f"{name} - {col} Boxplot")
        plt.ylabel(col)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                plots_folder,
                f"{name.lower()}_{col.lower()}_boxplot.png"
            ),
            dpi=300
        )
        plt.close()

print(f"\nPlots saved to: {plots_folder}")

# Save filtered Residential datasets
residential_listings.to_csv(
    os.path.join(folder_path, "residential_listings.csv"),
    index=False
)

residential_sold.to_csv(
    os.path.join(folder_path, "residential_sold.csv"),
    index=False
)