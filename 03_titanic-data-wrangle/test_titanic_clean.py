import pandas as pd
from titanic_clean import clean_titanic
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

def load_raw():
    return pd.read_csv(URL)

def test_clean_drops_cabin_ticket():
    df = load_raw()
    median_age = df["Age"].median() # Считаем медиану для теста
    cleaned = clean_titanic(df, median_age) # Передаем два аргумента
    assert "Cabin" not in cleaned.columns
    assert "Ticket" not in cleaned.columns

def test_clean_no_missing_age_or_embarked():
    df = load_raw()
    median_age = df["Age"].median()
    cleaned = clean_titanic(df, median_age)
    assert cleaned["Age"].isna().sum() == 0
    assert cleaned["Embarked"].isna().sum() == 0

def test_clean_shape():
    df = load_raw()
    median_age = df["Age"].median()
    cleaned = clean_titanic(df, median_age)
    assert cleaned.shape == (889, 10)