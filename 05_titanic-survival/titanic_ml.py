import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# ==============================================================================
# Exercise 0: Константы проекта
# ==============================================================================
# RANDOM_STATE фиксирует генератор случайных чисел для полной воспроизводимости
# результатов (разделение данных, инициализация моделей и деревьев решений).
RANDOM_STATE = 42
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


# ==============================================================================
# Exercise 1: Загрузка и предобработка данных (Cleaning)
# ==============================================================================
def load_and_clean(url: str = DATA_URL) -> pd.DataFrame:
    """
    Загружает датасет Титаника по URL и выполняет базовую очистку данных:
    1. Загрузка CSV через pandas.
    2. Удаление столбцов 'Cabin' (слишком много пропусков) и 'Ticket' (высокая кардинальность).
    3. Удаление строк с пропущенным портом посадки 'Embarked' (всего 2 строки).
    4. Заполнение пропусков в столбце 'Age' медианным значением (устойчиво к выбросам).
    5. Сброс индексов датафрейма для непрерывной индексации.
    """
    df = pd.read_csv(url)
    
    # Удаляем неинформативные признаки
    df = df.drop(columns=["Cabin", "Ticket"])
    
    # Удаляем единичные строки с неизвестным портом посадки
    df = df.dropna(subset=["Embarked"])
    
    # Заполняем пропущенный возраст медианой по выборке
    median_age = df["Age"].median()
    df["Age"] = df["Age"].fillna(median_age)
    
    # Сбрасываем индексы после удаления строк
    df = df.reset_index(drop=True)
    return df


# ==============================================================================
# Exercise 2: Базовый набор признаков (Baseline Features)
# ==============================================================================
def make_baseline_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Формирует минимальный baseline-набор числовых признаков без предобработки:
    - Признаки (X): ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    - Целевая переменная (y): 'Survived'
    """
    X = df[["Pclass", "Age", "SibSp", "Parch", "Fare"]].copy()
    y = df["Survived"].copy()
    return X, y


# ==============================================================================
# Exercise 3: Генерация признаков (Feature Engineering)
# ==============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создает новые информативные признаки и кодирует категориальные переменные:
    1. Sex_male: бинарное кодирование пола (1 = male, 0 = female).
    2. Embarked_*: one-hot кодирование порта посадки (C, Q, S).
    3. FamilySize: размер семьи пассажира на борту (SibSp + Parch + сам пассажир).
    4. IsAlone: индикатор путешествия в одиночку (1, если FamilySize == 1).
    5. AgeBand: дискретизация возраста на 5 категорий ([0, 12, 18, 35, 60, 100]).
    6. FareBand: разбиение стоимости билета на 4 квантиля (pd.qcut).
    7. Удаление сырых и неиспользуемых колонок: PassengerId, Name, Sex, Embarked, Survived.
    Итоговый датафрейм содержит ровно 13 спроектированных признаков.
    """
    featured = df.copy()

    # 1. Бинарный признак пола
    featured["Sex_male"] = (featured["Sex"] == "male").astype(int)

    # 2. One-hot кодирование порта посадки (Embarked)
    embarked_dummies = pd.get_dummies(
        featured["Embarked"], prefix="Embarked", dtype=int
    )
    featured = pd.concat([featured, embarked_dummies], axis=1)

    # 3. Размер семьи и признак путешествия в одиночку
    featured["FamilySize"] = featured["SibSp"] + featured["Parch"] + 1
    featured["IsAlone"] = (featured["FamilySize"] == 1).astype(int)

    # 4. Возрастные интервалы (AgeBand):
    # 0: дети (0-12), 1: подростки (12-18), 2: молодые взрослые (18-35),
    # 3: средний возраст (35-60), 4: пожилые (60-100)
    featured["AgeBand"] = pd.cut(
        featured["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=[0, 1, 2, 3, 4],
    ).astype(int)

    # 5. Квантильные интервалы стоимости билета (FareBand): 4 равные группы по стоимости
    featured["FareBand"] = pd.qcut(
        featured["Fare"],
        q=4,
        labels=[0, 1, 2, 3],
        duplicates="drop",
    ).astype(int)

    # 6. Удаление исходных нечисловых столбцов, идентификаторов и целевой переменной
    cols_to_drop = ["PassengerId", "Name", "Sex", "Embarked", "Survived"]
    cols_to_drop = [c for c in cols_to_drop if c in featured.columns]
    featured = featured.drop(columns=cols_to_drop)

    return featured


def main() -> None:
    # --------------------------------------------------------------------------
    # 1. Загрузка и очистка данных
    # --------------------------------------------------------------------------
    df = load_and_clean()
    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Cleaned columns: {list(df.columns)}")
    print(f"Missing values count: {df.isna().sum().sum()}")
    print(f"Target distribution:\n{df['Survived'].value_counts().to_dict()}\n")

    # --------------------------------------------------------------------------
    # 2. Обучение Baseline-моделей
    # --------------------------------------------------------------------------
    X_base, y = make_baseline_features(df)
    
    # Стратифицированное разбиение (80% train / 20% test) для сохранения баланса классов
    X_train_b, X_test_b, y_train, y_test = train_test_split(
        X_base, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Baseline train shape: {X_train_b.shape}, test shape: {X_test_b.shape}")

    # Для линейных моделей (LogisticRegression) необходимо масштабирование признаков.
    # Scaler обучается только на тренировочных данных (fit_transform), чтобы избежать data leakage.
    scaler_b = StandardScaler()
    X_train_b_scaled = scaler_b.fit_transform(X_train_b)
    X_test_b_scaled = scaler_b.transform(X_test_b)

    # Обучение Baseline LogisticRegression на масштабированных данных
    lr_base = LogisticRegression(max_iter=10000, random_state=RANDOM_STATE)
    lr_base.fit(X_train_b_scaled, y_train)
    logreg_base_acc = accuracy_score(y_test, lr_base.predict(X_test_b_scaled))

    # Обучение Baseline RandomForestClassifier на исходных (немасштабированных) данных,
    # так как деревья решений инвариантны к монотонному масштабированию признаков.
    rf_base = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_base.fit(X_train_b, y_train)
    rf_base_acc = accuracy_score(y_test, rf_base.predict(X_test_b))

    print(f"Baseline LogisticRegression accuracy: {logreg_base_acc:.4f}")
    print(f"Baseline RandomForestClassifier accuracy: {rf_base_acc:.4f}")

    # Сводная таблица baseline-моделей
    baseline_summary = pd.DataFrame({
        "model": ["LogisticRegression", "RandomForestClassifier"],
        "baseline_accuracy": [logreg_base_acc, rf_base_acc],
    })
    print("\nBaseline Models Summary:")
    print(baseline_summary.round(4).to_string(index=False))
    print()

    # --------------------------------------------------------------------------
    # 3. Feature Engineering (Генерация признаков)
    # --------------------------------------------------------------------------
    X_eng = engineer_features(df)
    print(f"Engineered features shape: {X_eng.shape}")
    print(f"Engineered columns ({len(X_eng.columns)}): {list(X_eng.columns)}")
    print(f"Sex_male=1 count: {(X_eng['Sex_male'] == 1).sum()}")
    print(f"IsAlone=1 count: {(X_eng['IsAlone'] == 1).sum()}")
    print(f"AgeBand=2 count: {(X_eng['AgeBand'] == 2).sum()}")
    print(f"AgeBand value counts: {X_eng['AgeBand'].value_counts().to_dict()}\n")

    # --------------------------------------------------------------------------
    # 4. Обучение моделей на спроектированных признаках (Engineered Models)
    # --------------------------------------------------------------------------
    # Используем те же параметры split (random_state и stratify) для честного сравнения
    X_train_e, X_test_e, _, _ = train_test_split(
        X_eng, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Engineered train shape: {X_train_e.shape}, test shape: {X_test_e.shape}")

    # Масштабирование признаков для LogisticRegression
    scaler_e = StandardScaler()
    X_train_e_scaled = scaler_e.fit_transform(X_train_e)
    X_test_e_scaled = scaler_e.transform(X_test_e)

    # Обучение улучшенной LogisticRegression
    lr_eng = LogisticRegression(max_iter=10000, random_state=RANDOM_STATE)
    lr_eng.fit(X_train_e_scaled, y_train)
    logreg_eng_acc = accuracy_score(y_test, lr_eng.predict(X_test_e_scaled))

    # Обучение улучшенного RandomForestClassifier
    rf_eng = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_eng.fit(X_train_e, y_train)
    rf_eng_pred = rf_eng.predict(X_test_e)
    rf_eng_acc = accuracy_score(y_test, rf_eng_pred)

    print(f"Engineered LogisticRegression accuracy: {logreg_eng_acc:.4f}")
    print(f"Engineered RandomForestClassifier accuracy: {rf_eng_acc:.4f}\n")

    # Детальный отчет о метриках (precision, recall, f1-score) для обоих классов
    print("Engineered Random Forest Classification Report:")
    print(classification_report(y_test, rf_eng_pred, target_names=["died", "survived"]))

    # Анализ важности признаков (Feature Importances) из обученного случайного леса
    importances = pd.Series(rf_eng.feature_importances_, index=X_eng.columns).sort_values(
        ascending=False
    )
    print("Top 5 RF Feature Importances:")
    print(f"{importances.head(5)}\n")

    # --------------------------------------------------------------------------
    # 5. Сравнение моделей (Comparison)
    # --------------------------------------------------------------------------
    comparison = pd.DataFrame({
        "model": ["LogisticRegression", "RandomForestClassifier"],
        "baseline_accuracy": [logreg_base_acc, rf_base_acc],
        "engineered_accuracy": [logreg_eng_acc, rf_eng_acc],
    })
    # Вычисляем прирост точности от добавления новых признаков (delta)
    comparison["delta"] = (
        comparison["engineered_accuracy"] - comparison["baseline_accuracy"]
    )
    print("Model Comparison:")
    print(comparison.round(4).to_string(index=False))
    print()

    # Определение лучшей модели
    best_idx = comparison["engineered_accuracy"].idxmax()
    best_model = comparison.loc[best_idx, "model"]
    best_acc = comparison.loc[best_idx, "engineered_accuracy"]

    print("Run Summary:")
    print(
        f"LogisticRegression improved from {logreg_base_acc:.4f} to {logreg_eng_acc:.4f} "
        f"(delta: +{comparison.loc[0, 'delta']:.4f})"
    )
    print(
        f"RandomForestClassifier improved from {rf_base_acc:.4f} to {rf_eng_acc:.4f} "
        f"(delta: +{comparison.loc[1, 'delta']:.4f})"
    )
    print(f"Best model: {best_model} with accuracy {best_acc:.4f}\n")

    # --------------------------------------------------------------------------
    # 6. Сохранение итогового отчета в summary.txt
    # --------------------------------------------------------------------------
    with open("summary.txt", "w") as f:
        f.write("Titanic ML Raid — Summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(comparison.round(4).to_string(index=False) + "\n\n")
        f.write(f"Best model: {comparison.loc[best_idx, 'model']}\n")
        f.write(
            f"Best test accuracy: {comparison.loc[best_idx, 'engineered_accuracy']:.4f}\n"
        )


if __name__ == "__main__":
    main()
