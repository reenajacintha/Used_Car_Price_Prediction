
import pandas as pd

# Load the used car dataset
df = pd.read_csv("data/CAR DETAILS FROM CAR DEKHO.csv")

# Display first 5 records
print("FIRST 5 RECORDS")
print(df.head())

# Display dataset dimensions
print("\nDATASET SHAPE")
print(df.shape)

# Display column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())

# Display dataset information
print("\nDATASET INFORMATION")
df.info()

# Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Check missing value percentages
print("\nMISSING VALUE PERCENTAGE")
print((df.isnull().sum() / len(df)) * 100)

# Check duplicate records
print("\nDUPLICATE RECORDS")

duplicate_count = df.duplicated().sum()

print("Total duplicate rows:", duplicate_count)

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Verify duplicates are removed
print("\nAFTER REMOVING DUPLICATES")

print("Remaining duplicate rows:", df.duplicated().sum())

print("Updated dataset shape:", df.shape)

# Check numerical statistics
print("\nNUMERICAL FEATURE STATISTICS")
print(df.describe())

# Check invalid values
print("\nINVALID VALUES")

print("Invalid selling prices:", (df["selling_price"] <= 0).sum())
print("Invalid kilometres:", (df["km_driven"] <= 0).sum())
print("Invalid manufacturing years:", (df["year"] <= 0).sum())

# Remove invalid car records
df = df[
    (df["selling_price"] > 0) &
    (df["km_driven"] > 0) &
    (df["year"] > 0)
].copy()

print("\nAFTER REMOVING INVALID RECORDS")
print("Updated dataset shape:", df.shape)

print("Invalid selling prices:", (df["selling_price"] <= 0).sum())
print("Invalid kilometres:", (df["km_driven"] <= 0).sum())
print("Invalid manufacturing years:", (df["year"] <= 0).sum())
# Inspect categorical columns
categorical_columns = ["fuel", "seller_type", "transmission", "owner"]

for column in categorical_columns:
    print(f"\n{column.upper()} VALUE COUNTS")
    print(df[column].value_counts())
    # Inspect car names
print("\nTOTAL UNIQUE CAR NAMES:")
print(df["name"].nunique())

print("\nTOP 20 CAR MODELS:")
print(df["name"].value_counts().head(20))
# Save the cleaned dataset
df.to_csv("data/cleaned_car_data.csv", index=False)

print("\nCLEANED DATASET SAVED SUCCESSFULLY")
