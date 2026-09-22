import pandas as pd
import glob
import os

# Ask for the folder containing the monthly CSV files
folder_path = input("Enter the path to the folder containing the monthly CSV files: ")

csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Separate Listings and Sold files
listings_files = [
    file for file in csv_files
    if "listing" in os.path.basename(file).lower()
]

sold_files = [
    file for file in csv_files
    if "sold" in os.path.basename(file).lower()
]

# Check that files were found
print(f"Number of Listings files found: {len(listings_files)}")
print(f"Number of Sold files found: {len(sold_files)}")

# Add up all rows from the individual files to verify against the concatenated DataFrames
total_listings_rows = sum(
    len(pd.read_csv(file, low_memory=False))
    for file in listings_files
)

total_sold_rows = sum(
    len(pd.read_csv(file, low_memory=False))
    for file in sold_files
)

print(f"Total rows from individual Listings files: {total_listings_rows}")
print(f"Total rows from individual Sold files: {total_sold_rows}")

# Read and concatenate Listings files
listings_df = pd.concat(
    (pd.read_csv(file, low_memory=False) for file in listings_files),
    ignore_index=True
)

# Read and concatenate Sold files
sold_df = pd.concat(
    (pd.read_csv(file, low_memory=False) for file in sold_files),
    ignore_index=True
)

# Print row counts after concatenation
print(
    f"Total rows in Listings DataFrame before Residential filter: "
    f"{len(listings_df)}"
)

print(
    f"Total rows in Sold DataFrame before Residential filter: "
    f"{len(sold_df)}"
)

# Filter dataframes to only include Residential properties
listings_df = listings_df[
    listings_df["PropertyType"].str.contains(
        "Residential",
        case=False,
        na=False
    )
]

sold_df = sold_df[
    sold_df["PropertyType"].str.contains(
        "Residential",
        case=False,
        na=False
    )
]

# Print row counts after Residential filter
print(
    f"Total rows in Listings DataFrame after Residential filter: "
    f"{len(listings_df)}"
)

print(
    f"Total rows in Sold DataFrame after Residential filter: "
    f"{len(sold_df)}"
)

# Save the filtered DataFrames to new CSV files
listings_output_path = os.path.join(folder_path, "filtered_listings.csv")
sold_output_path = os.path.join(folder_path, "filtered_sold.csv")

listings_df.to_csv(listings_output_path, index=False)
sold_df.to_csv(sold_output_path, index=False)

print(f"Filtered Listings saved to: {listings_output_path}")
print(f"Filtered Sold saved to: {sold_output_path}")
