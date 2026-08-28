# 🌴 Cali-Scope: California Housing Exploratory Data Analysis

[English Version below]

---

## 🇷🇺 Обзор Проекта (Russian)

Данный проект представляет собой глубокий исследовательский анализ данных (EDA) цен на жилье в Калифорнии. Мы проанализировали более 20 000 районов, чтобы выявить ключевые факторы, влияющие на стоимость недвижимости, и визуализировать географические паттерны.

### 🛠 Стек технологий
- **Pandas & NumPy**: Обработка и очистка данных.
- **Matplotlib & Seaborn**: Создание сложных визуализаций (`subplot_mosaic`).
- **Pytest**: Автоматизированное тестирование пайплайна очистки.

### 🚀 Запуск проекта
1. Создайте и активируйте окружение:
python3 -m venv venv
source venv/bin/activate
2. Установите зависимости:
pip install pandas numpy matplotlib seaborn pytest
3. Запустите тесты:
python -m pytest test_housing_eda.py -v
4. Запустите анализ:
python3 housing_eda.py

### 📊 Ключевые инсайты
1. **Качество данных**: Обнаружено 207 пропусков в `total_bedrooms` (заполнены медианой) и искусственный "потолок" цен на уровне **$500,001** (965 строк). Это критично для будущего обучения моделей.
2. **Главный фактор**: Доход населения (`median_income`) имеет сильнейшую корреляцию с ценой (**0.6881**).
3. **География**: Самое дорогое жилье сосредоточено на побережье. Жилье в материковой части (**Inland**) в среднем в 4 раза дешевле, чем на островах или в элитных заливах.
4. **Избыточность**: Колонки комнат, спален и населения сильно коррелируют (>0.9), что требует их объединения в будущем (например, "комнат на человека").

---

## 🇺🇸 Project Overview (English)

Cali-Scope is a comprehensive end-to-end EDA pipeline for the California Housing dataset. We analyzed 20,640 census blocks to diagnose data quality and identify the primary drivers of real estate value.

### 📊 Key Findings & Diagnostics
- **Imputation**: 207 missing values in `total_bedrooms` were handled via **median imputation** to ensure robustness against skewed distributions.
- **Data Censoring**: Identified a target cap at **$500,001**. This "right-censoring" is a major data quality issue that will distort linear models if not handled properly.
- **Skewness**: Most features (population, rooms) are heavily right-skewed (Skew > 4.0), indicating high urban density in specific pockets.
- **Predictive Power**: `median_income` is the strongest predictor (r = 0.688). 
- **Geographic Premium**: A clear price gradient exists from **Inland** (Median: $108.5k) to **Island** (Median: $414.7k).

### 🖼 Visualization Strategy
We utilized `plt.subplot_mosaic` to create a 6-panel "Chart Pack" (`plots/ex05_chartpack.png`) that combines:
1. **Geographic Scatter**: Latitude/Longitude plotted with price (color) and population (size).
2. **Distributions**: Histograms for income and house value.
3. **Categorical Analysis**: Boxplots by ocean proximity.
4. **Statistical Analysis**: Annotated correlation heatmaps.

### 🧪 Quality Assurance
The project includes a `pytest` suite ensuring:
- **Zero missing values** after cleaning.
- **Shape preservation** (no rows lost).
- **Idempotency** (repeated cleaning doesn't change the data).

---

## ⚖ Conclusion / Вывод
Wealth and location (proximity to the coast) are the dominant factors. However, the price capping at $500,001 is a "trap" for future Machine Learning models — they will systematically under-predict expensive homes unless this is addressed.