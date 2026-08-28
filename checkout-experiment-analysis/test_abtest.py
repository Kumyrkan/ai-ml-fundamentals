import numpy as np
import pytest
from abtest import generate_data

def test_generate_data_shapes():
    # Проверяем, что все массивы имеют нужную длину (1000)
    data = generate_data(seed=42)
    for key in data:
        assert data[key].shape == (1000,), f"Массив {key} имеет неверную форму"

def test_generate_data_reproducible():
    # Проверяем, что один и тот же сид дает идентичные результаты
    data1 = generate_data(seed=42)
    data2 = generate_data(seed=42)
    for key in data1:
        # np.array_equal сравнивает массивы поэлементно
        assert np.array_equal(data1[key], data2[key]), f"Массивы {key} не идентичны для сида 42"

def test_generate_data_different_seeds():
    # Проверяем, что разные сиды дают разные данные
    data1 = generate_data(seed=42)
    data2 = generate_data(seed=1)
    for key in data1:
        # Мы ожидаем, что данные БУДУТ отличаться
        assert not np.array_equal(data1[key], data2[key]), f"Массивы {key} одинаковы для разных сидов"