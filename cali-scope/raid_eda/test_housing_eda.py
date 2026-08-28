import pytest
import pandas as pd
import numpy as np
from housing_eda import clean_housing, URL

def test_clean_no_missing():
    df = pd.read_csv(URL)
    cleaned = clean_housing(df)
    assert cleaned.isna().sum().sum() == 0

def test_clean_preserves_shape():
    df = pd.read_csv(URL)
    cleaned = clean_housing(df)
    assert cleaned.shape == df.shape

def test_clean_idempotent():
    df = pd.read_csv(URL)
    first_clean = clean_housing(df)
    second_clean = clean_housing(first_clean)
    pd.testing.assert_frame_equal(first_clean, second_clean)