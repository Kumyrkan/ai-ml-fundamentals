# 🚢 Titanic Survival Prediction: ML Raid

[🇷🇺 Описание на русском](#russian-description)

## 📌 Project Overview
This project is a comprehensive Machine Learning raid focused on predicting passenger survival on the Titanic. The primary goal was to demonstrate how **Feature Engineering** and proper **Data Preprocessing** can significantly boost model performance compared to a baseline.

The project implements a full pipeline: from raw data loading and cleaning to advanced feature extraction and model comparison (Logistic Regression vs. Random Forest).

## 🎯 Learning Objectives
*   **Data Cleaning:** Handling missing values (median imputation) and dropping irrelevant features.
*   **Baseline Modeling:** Establishing a performance floor using minimal numeric features.
*   **Feature Engineering:** Creating new meaningful features: `FamilySize`, `IsAlone`, `AgeBand`, `FareBand`, and One-Hot Encoding for categorical variables.
*   **Performance Analysis:** Measuring the "Delta" (improvement) for each model after engineering.
*   **Reproducibility:** Ensuring 100% deterministic results using `RANDOM_STATE = 42`.

## 🛠 Tech Stack
*   **Language:** Python 3.10+
*   **Libraries:** Pandas, NumPy, Scikit-learn
*   **Models:** Logistic Regression, Random Forest Classifier

## 📊 Key Results (Example)
After engineering, the models showed significant accuracy improvements:
*   **Logistic Regression:** Baseline ~70.8% → Engineered **~81.5% (+10.7%)**
*   **Random Forest:** Baseline ~65.2% → Engineered **~78.7% (+13.5%)**

## ⚙️ How to Run
```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install pandas numpy scikit-learn
   python3 titanic_ml.py
Check the generated summary.txt for the final audit report.

<a name="russian-description"></a>

## 🇷🇺 Описание проекта (Русский)
Данный проект — это практический "рейд" по машинному обучению, посвященный задаче классификации выживаемости пассажиров Титаника. Основной акцент сделан на Feature Engineering и сравнении алгоритмов.

### 🚀 Основные этапы
Очистка данных: Удаление неинформативных колонок (Cabin, Ticket), заполнение пропусков в возрасте медианой и фильтрация строк.

Baseline: Обучение моделей на "сырых" числовых признаках для фиксации начальной точности.

Feature Engineering: Создание новых признаков:

FamilySize: сумма родственников на борту.

IsAlone: бинарный признак одиночного путешествия.

AgeBand & FareBand: группировка возраста и стоимости билета в категории (бининг).

Sex_male & Embarked: кодирование категориальных данных.

Сравнение: Оценка прироста точности (accuracy) для логистической регрессии и случайного леса.

### 📈 Итоги
Лучшая модель: Logistic Regression (точность ~81.46%).

Воспроизводимость: Проект полностью детерминирован за счет фиксации random_state=42.

Отчетность: Результаты автоматически сохраняются в файл summary.txt.

### 📁 Структура папки
titanic_ml.py — основной скрипт с кодом.

summary.txt — сгенерированный отчет с метриками.

requirements.txt — список зависимостей.