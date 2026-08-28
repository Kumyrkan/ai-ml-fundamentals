import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ssl

# Фикс SSL для macOS
ssl._create_default_https_context = ssl._create_unverified_context

URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

def load_and_inspect():
    df = pd.read_csv(URL)
    print("="*50)
    print("СТАДИЯ 1: ПЕРВИЧНЫЙ ОСМОТР ДАННЫХ")
    print("="*50)
    print(f"РАЗМЕРНОСТЬ: {df.shape[0]} строк, {df.shape[1]} колонок")
    
    print("\nТИПЫ ДАННЫХ:")
    print(df.dtypes)
    
    print("\nПРОПУЩЕННЫЕ ЗНАЧЕНИЯ (ДО ОЧИСТКИ):")
    missing = df.isna().sum()
    print(missing[missing > 0])
    
    print("\nРАСПРЕДЕЛЕНИЕ ПО БЛИЗОСТИ К ОКЕАНУ:")
    print(df["ocean_proximity"].value_counts())
    
    print("\nОСНОВНАЯ СТАТИСТИКА (ОБРАТИТЕ ВНИМАНИЕ НА MAX ЗНАЧЕНИЯ):")
    print(df.describe().round(2))
    return df

def clean_housing(df):
    df_clean = df.copy()
    median_val = df_clean["total_bedrooms"].median()
    df_clean["total_bedrooms"] = df_clean["total_bedrooms"].fillna(median_val)
    return df_clean

def run_univariate(df):
    os.makedirs("plots", exist_ok=True)
    # 3x3 Grid of Histograms
    df.hist(bins=50, figsize=(15, 12))
    plt.suptitle("Univariate Analysis: Feature Distributions", fontsize=16)
    plt.savefig("plots/ex02_univariate.png")
    plt.close()

    print("\n" + "="*50)
    print("СТАДИЯ 2: ОДНОМЕРНЫЙ АНАЛИЗ (SKEWNESS)")
    print("="*50)
    print(df.skew(numeric_only=True).round(2))
    
    capped = (df["median_house_value"] == 500001).sum()
    print(f"\nВНИМАНИЕ: Обнаружен искусственный порог цены ($500,001)")
    print(f"Количество затронутых строк: {capped}")


def run_bivariate(df):
    # Correlation with target
    corrs = df.corr(numeric_only=True)["median_house_value"].abs().sort_values(ascending=False)
    print("\nCorrelation with Median House Value:\n", corrs)

    # Scatter: Income vs Value (Sample 2000)
    plt.figure(figsize=(10, 6))
    sample = df.sample(2000, random_state=42)
    sns.scatterplot(data=sample, x="median_income", y="median_house_value", hue="ocean_proximity")
    plt.title("Income vs House Value (Sample 2000)")
    plt.savefig("plots/ex03_income_value.png")
    plt.close()

    # Boxplot with explicit order
    order = ["INLAND", "<1H OCEAN", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="ocean_proximity", y="median_house_value", order=order)
    plt.title("House Value by Ocean Proximity")
    plt.savefig("plots/ex03_value_by_proximity.png")
    plt.close()

    print("\n" + "="*50)
    print("СТАДИЯ 3: ДВУМЕРНЫЙ АНАЛИЗ (КОРРЕЛЯЦИЯ С ЦЕНОЙ)")
    print("="*50)
    corrs = df.corr(numeric_only=True)["median_house_value"].abs().sort_values(ascending=False)
    print(corrs)
    
    print("\nМЕДИАННАЯ СТОИМОСТЬ ПО КАТЕГОРИЯМ БЛИЗОСТИ:")
    print(df.groupby("ocean_proximity")["median_house_value"].median().sort_values())

def run_multivariate(df):
    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", vmin=-1, vmax=1, square=True)
    plt.title("Correlation Heatmap")
    plt.savefig("plots/ex04_corr.png")
    plt.close()

    # Pairplot (Sample 1000)
    top_features = ["median_income", "total_rooms", "housing_median_age", "latitude", "median_house_value"]
    sample_pair = df.sample(1000, random_state=42)
    sns.pairplot(sample_pair[top_features + ["ocean_proximity"]], hue="ocean_proximity")
    plt.savefig("plots/ex04_pairplot.png")
    plt.close()

def run_geo_mosaic(df):
    # Setup Mosaic
    layout = """
        AAB
        AAC
        DEF
    """
    fig, axd = plt.subplot_mosaic(layout, figsize=(18, 14))
    
    # A: Geographic Scatter (The centerpiece)
    sc = axd['A'].scatter(df["longitude"], df["latitude"], alpha=0.4,
                          s=df["population"]/100, label="population",
                          c=df["median_house_value"], cmap="plasma")
    axd['A'].set_title("Geographic Distribution: Price & Population")
    plt.colorbar(sc, ax=axd['A'], label="Median House Value")
    
    # B: House Value Hist
    sns.histplot(df["median_house_value"], bins=50, ax=axd['B'], color='teal')
    axd['B'].set_title("House Value Distribution")

    # C: Income Hist
    sns.histplot(df["median_income"], bins=50, ax=axd['C'], color='orange')
    axd['C'].set_title("Income Distribution")

    # D: Income vs Value
    sns.scatterplot(data=df.sample(2000, random_state=42), x="median_income", 
                    y="median_house_value", hue="ocean_proximity", ax=axd['D'], legend=False)
    axd['D'].set_title("Income vs Value")

    # E: Boxplot
    order = ["INLAND", "<1H OCEAN", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
    sns.boxplot(data=df, x="ocean_proximity", y="median_house_value", order=order, ax=axd['E'])
    axd['E'].tick_params(axis='x', rotation=45)
    axd['E'].set_title("Value by Proximity")

    # F: Correlation Heatmap (Top 4)
    top_4_cols = ["median_income", "total_rooms", "housing_median_age", "latitude"]
    sns.heatmap(df[top_4_cols + ["median_house_value"]].corr(), annot=True, cmap="coolwarm", ax=axd['F'], cbar=False)
    axd['F'].set_title("Top Features Correlation")

    fig.suptitle("California Housing EDA — Chart Pack", fontsize=20, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/ex05_chartpack.png")
    plt.close()

    # Individual Geo Plot
    plt.figure(figsize=(10, 7))
    plt.scatter(df["longitude"], df["latitude"], alpha=0.4, s=df["population"]/100,
                c=df["median_house_value"], cmap="plasma")
    plt.title("Geographic Housing Map")
    plt.savefig("plots/ex05_geo.png")
    plt.close()

def main():
    df = load_and_inspect()
    df_clean = clean_housing(df)
    run_univariate(df_clean)
    run_bivariate(df_clean)
    run_multivariate(df_clean)
    run_geo_mosaic(df_clean)

if __name__ == "__main__":
    main()