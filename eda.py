import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/cleaned_car_data.csv")

# Selling price statistics
print("SELLING PRICE STATISTICS")
print(df["selling_price"].describe())

# Selling price distribution
plt.figure(figsize=(8, 5))
plt.hist(df["selling_price"], bins=30, edgecolor="black")

plt.title("Distribution of Used Car Selling Prices")
plt.xlabel("Selling Price (INR)")
plt.ylabel("Number of Cars")

plt.savefig("selling_price_distribution.png", dpi=300, bbox_inches="tight")
plt.show()
# Distribution of car manufacturing years
plt.figure(figsize=(8, 5))

plt.hist(df["year"], bins=20, edgecolor="black")

plt.title("Distribution of Car Manufacturing Years")
plt.xlabel("Manufacturing Year")
plt.ylabel("Number of Cars")

plt.savefig("manufacturing_year_distribution.png",
            dpi=300, bbox_inches="tight")

plt.show()
# Distribution of kilometres driven
plt.figure(figsize=(8, 5))

plt.hist(df["km_driven"], bins=30, edgecolor="black")

plt.title("Distribution of Kilometres Driven")
plt.xlabel("Kilometres Driven")
plt.ylabel("Number of Cars")

plt.savefig("km_driven_distribution.png",
            dpi=300, bbox_inches="tight")

plt.show()
# Fuel type distribution
plt.figure(figsize=(8, 5))

df["fuel"].value_counts().plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Number of Cars by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Number of Cars")
plt.xticks(rotation=0)

plt.savefig(
    "fuel_type_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# Transmission type distribution
plt.figure(figsize=(7, 5))

df["transmission"].value_counts().plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Number of Cars by Transmission Type")
plt.xlabel("Transmission Type")
plt.ylabel("Number of Cars")
plt.xticks(rotation=0)

plt.savefig(
    "transmission_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# Owner type distribution
plt.figure(figsize=(9, 5))

df["owner"].value_counts().plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Number of Cars by Owner Type")
plt.xlabel("Owner Type")
plt.ylabel("Number of Cars")
plt.xticks(rotation=30, ha="right")

plt.savefig(
    "owner_type_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# Relationship between manufacturing year and selling price
plt.figure(figsize=(9, 5))

plt.scatter(
    df["year"],
    df["selling_price"],
    alpha=0.5
)

plt.title("Manufacturing Year vs Selling Price")
plt.xlabel("Manufacturing Year")
plt.ylabel("Selling Price (INR)")

plt.savefig(
    "year_vs_selling_price.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# Relationship between kilometres driven and selling price
plt.figure(figsize=(9, 5))

plt.scatter(
    df["km_driven"],
    df["selling_price"],
    alpha=0.5
)

plt.title("Kilometres Driven vs Selling Price")
plt.xlabel("Kilometres Driven")
plt.ylabel("Selling Price (INR)")

plt.savefig(
    "km_vs_selling_price.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# Average selling price by fuel type
fuel_prices = df.groupby("fuel")["selling_price"].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 5))

fuel_prices.plot(kind="bar", edgecolor="black")

plt.title("Average Selling Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Average Selling Price (INR)")
plt.xticks(rotation=0)

plt.savefig(
    "fuel_vs_selling_price.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# Average selling price by transmission type
transmission_prices = (
    df.groupby("transmission")["selling_price"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(7, 5))

transmission_prices.plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Average Selling Price by Transmission Type")
plt.xlabel("Transmission Type")
plt.ylabel("Average Selling Price (INR)")
plt.xticks(rotation=0)

plt.savefig(
    "transmission_vs_selling_price.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# Correlation analysis
numerical_columns = ["year", "km_driven", "selling_price"]

correlation_matrix = df[numerical_columns].corr()

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.savefig(
    "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# Feature Engineering - Calculate Car Age

current_year = 2026

df["car_age"] = current_year - df["year"]

print("\nCAR AGE FEATURE")
print(df[["year", "car_age"]].head())

print("\nCAR AGE STATISTICS")
print(df["car_age"].describe())
# Car age vs selling price
plt.figure(figsize=(9, 5))

plt.scatter(
    df["car_age"],
    df["selling_price"],
    alpha=0.5
)

plt.title("Car Age vs Selling Price")
plt.xlabel("Car Age (Years)")
plt.ylabel("Selling Price (INR)")

plt.savefig(
    "car_age_vs_selling_price.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# Outlier analysis - Selling Price
plt.figure(figsize=(8, 5))

sns.boxplot(x=df["selling_price"])

plt.title("Outlier Analysis of Selling Price")
plt.xlabel("Selling Price (INR)")

plt.savefig(
    "selling_price_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# Identify outliers using IQR method

Q1 = df["selling_price"].quantile(0.25)
Q3 = df["selling_price"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

outliers = df[
    (df["selling_price"] < lower_limit) |
    (df["selling_price"] > upper_limit)
]

print("\nOUTLIER ANALYSIS")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Limit:", lower_limit)
print("Upper Limit:", upper_limit)
print("Total Outliers:", len(outliers))
# Display highest selling price records
print("\nTOP 10 HIGHEST PRICED CARS")

print(
    df.nlargest(10, "selling_price")[
        ["name", "year", "km_driven", "selling_price"]
    ]
)
# Extract car brand from car name
df["brand"] = df["name"].str.split().str[0]

print("\nCAR BRAND DISTRIBUTION")
print(df["brand"].value_counts().head(15))
print("\nALL UNIQUE BRANDS:")
print(sorted(df["brand"].unique()))

print("\nTOTAL UNIQUE BRANDS:")
print(df["brand"].nunique())