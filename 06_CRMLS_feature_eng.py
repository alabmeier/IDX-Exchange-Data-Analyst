import os
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

