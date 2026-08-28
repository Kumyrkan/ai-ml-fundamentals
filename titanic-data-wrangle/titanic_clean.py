import pandas as pd
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

def load_and_inspect():
    df = pd.read_csv(URL)
    # --- ТЗ требует эти принты ---
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())
    print("\nSex counts:\n", df["Sex"].value_counts())
    print("\nPclass counts:\n", df["Pclass"].value_counts())
    print("\nEmbarked counts:\n", df["Embarked"].value_counts())
    
    # Считаем медиану СРАЗУ, до всяких дропов (как просит ТЗ)
    median_age = df["Age"].median() 
    return df, median_age

def clean_titanic(df, median_age):
    df = df.drop(columns=["Cabin", "Ticket"])
    df = df.dropna(subset=["Embarked"])
    df["Age"] = df["Age"].fillna(median_age) # Используем ту самую 28.0
    df["Sex"] = df["Sex"].astype("category")
    df["Embarked"] = df["Embarked"].astype("category")
    df = df.reset_index(drop=True)
    return df

def engineer_features(df):
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["AgeBand"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "Adult", "Senior", "Elder"]
    )
    df["FareBand"] = pd.qcut(df["Fare"], q=4, labels=["Low", "Mid", "High", "VeryHigh"])
    return df

def analyze(df):
    print("\nSurvival by Sex and Class:")

    sex_class_survival = df.groupby(["Sex", "Pclass"], observed=True).agg(
        survival_rate=("Survived", "mean"),
        n=("Survived", "count")
    )
    print(sex_class_survival)

def merge_with_ports(df):
    ports = pd.DataFrame({
        "Embarked": ["S", "C", "Q"],
        "PortName": ["Southampton", "Cherbourg", "Queenstown"],
        "Country": ["England", "France", "Ireland"]
    })
    ports["Embarked"] = ports["Embarked"].astype("category")
    df = pd.merge(df, ports, on="Embarked", how="left")
    print("После объединения:", df.shape)
    print(df.groupby("PortName", observed=True)["Survived"].agg(
        survival_rate=("mean"), n=("size")).round(4))
    print(df.pivot_table(values="Survived", index="Country", columns="Sex", aggfunc="mean").round(4))
    return df

def main():
    df, median_age = load_and_inspect()
    df = clean_titanic(df, median_age)
    df = engineer_features(df)
    analyze(df)
    df = merge_with_ports(df)
    # Тут можно добавить финальные принты по странам

if __name__ == "__main__":
    main()