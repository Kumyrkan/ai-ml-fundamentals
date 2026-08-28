import numpy as np
from scipy import stats

def generate_data(seed=42):
    # Создаем генератор случайных чисел с фиксированным сидом.
    # default_rng(seed) гарантирует, что при каждом запуске мы получим 
    # одни и те же числа. Это основа воспроизводимости в науке о данных.
    rng = np.random.default_rng(seed)
    
    # Генерируем длительность сессий (непрерывная величина)
    # loc - среднее значение, scale - стандартное отклонение (разброс)
    control_duration = rng.normal(loc=8.0, scale=3.0, size=1000)
    treatment_duration = rng.normal(loc=8.5, scale=3.0, size=1000)
    
    # Генерируем конверсию (бинарная величина: 1 - купил, 0 - нет)
    # rng.random(1000) создает 1000 чисел от 0 до 1.
    # Если число меньше вероятности (p), то это успех (1).
    control_converted = (rng.random(1000) < 0.10).astype(int)
    treatment_converted = (rng.random(1000) < 0.13).astype(int)
    
    return {
        "control_duration": control_duration,
        "treatment_duration": treatment_duration,
        "control_converted": control_converted,
        "treatment_converted": treatment_converted
    }

def describe_continuous(name, x):
    """Описывает непрерывные данные (длительность сессии)"""
    n = len(x)
    mean = np.mean(x)
    # ddof=1 используется для оценки СТАНДАРТНОГО ОТКЛОНЕНИЯ выборки.
    # Это поправка Бесселя, она делает оценку более точной для малых групп.
    std = np.std(x, ddof=1) 
    median = np.median(x)
    q1 = np.percentile(x, 25) # 25% пользователей сидели меньше этого времени
    q3 = np.percentile(x, 75) # 75% пользователей сидели меньше этого времени
    
    print(f"\n--- {name} (Duration) ---")
    print(f"n: {n}, mean: {mean:.4f}, std: {std:.4f}")
    print(f"median: {median:.4f}, q1: {q1:.4f}, q3: {q3:.4f}")
    return mean

def describe_binary(name, x):
    """Описывает бинарные данные (конверсия)"""
    n = len(x)
    count = np.sum(x) # Сумма единиц — это количество покупок
    rate = np.mean(x) # Среднее от 0 и 1 — это и есть конверсия (доля)
    
    print(f"\n--- {name} (Conversion) ---")
    print(f"n: {n}, count: {count}, rate: {rate:.4f}")
    return rate

def check_assumptions(control, treatment):
    print("\n--- Assumption Checks (Normality & Variance) ---")
    
    # 1. Тест Шапиро-Уилкса на нормальность.
    # Нулевая гипотеза (H0): данные распределены нормально.
    # Если p-value > 0.05, мы НЕ отклоняем H0 (значит всё ок, данные нормальные).
    shapiro_ctrl = stats.shapiro(control)
    shapiro_trtm = stats.shapiro(treatment)
    
    print(f"Shapiro-Wilk (Control): p-value = {shapiro_ctrl.pvalue:.4f}")
    print(f"Shapiro-Wilk (Treatment): p-value = {shapiro_trtm.pvalue:.4f}")
    
    # 2. Тест Левене на равенство дисперсий (гомоскедастичность).
    # Нулевая гипотеза (H0): дисперсии в группах равны.
    # Если p-value > 0.05, значит разброс в группах примерно одинаковый.
    levene_test = stats.levene(control, treatment)
    print(f"Levene's Test: p-value = {levene_test.pvalue:.4f}")
    
    # Выбор теста на основе проверок:
    # По ТЗ мы используем Welch t-test (equal_var=False), так как он более робастный.
    print("\nDecision: Continuous metric uses Welch t-test; binary metric uses chi-squared 2x2.")

def run_hypothesis_tests(data):
    print("\n--- Hypothesis Testing ---")
    
    # 1. Welch t-test для Длительности
    # equal_var=False делает t-тест именно тестом Уэлча.
    t_stat, t_p = stats.ttest_ind(data['control_duration'], 
                                  data['treatment_duration'], 
                                  equal_var=False)
    
    # 2. Mann-Whitney U тест (непараметрический аналог)
    u_stat, u_p = stats.mannwhitneyu(data['control_duration'], 
                                     data['treatment_duration'])
    
    print(f"Duration Welch: t={t_stat:.4f}, p={t_p:.4f}, reject? {'Yes' if t_p < 0.05 else 'No'}")
    print(f"Duration MWU:   U={u_stat}, p={u_p:.4f}, reject? {'Yes' if u_p < 0.05 else 'No'}")
    
    # 3. Chi-squared для Конверсии
    # Сначала строим таблицу сопряженности (Contingency table)
    # [ [успехи_А, неуспехи_А], [успехи_Б, неуспехи_Б] ]
    con_success = np.sum(data['control_converted'])
    tr_success = np.sum(data['treatment_converted'])
    
    table = [
        [con_success, 1000 - con_success],
        [tr_success, 1000 - tr_success]
    ]
    
    chi2, chi2_p, _, _ = stats.chi2_contingency(table)
    print(f"\nConversion Chi-squared: chi2={chi2:.4f}, p={chi2_p:.4f}, reject? {'Yes' if chi2_p < 0.05 else 'No'}")
    
    return t_p, chi2_p

def calculate_effect_sizes(data):
    print("\n--- Effect Sizes & Confidence Intervals ---")
    
    # --- Длительность (Duration) ---
    c_dur = data['control_duration']
    t_dur = data['treatment_duration']
    
    # 1. Cohen's d (Размер эффекта)
    # Формула: (mean2 - mean1) / pooled_std
    n1, n2 = len(c_dur), len(t_dur)
    var1, var2 = np.var(c_dur, ddof=1), np.var(t_dur, ddof=1)
    # Считаем объединенное стандартное отклонение
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = (np.mean(t_dur) - np.mean(c_dur)) / pooled_std
    
    # 2. 95% Confidence Interval для разницы средних
    diff = np.mean(t_dur) - np.mean(c_dur)
    se_dur = np.sqrt(var1/n1 + var2/n2) # Стандартная ошибка
    ci_dur = [diff - 1.96 * se_dur, diff + 1.96 * se_dur]
    
    print(f"Duration Cohen's d: {d:.4f}")
    print(f"Duration 95% CI: [{ci_dur[0]:.4f}, {ci_dur[1]:.4f}]")
    
    # --- Конверсия (Conversion) ---
    p1 = np.mean(data['control_converted'])
    p2 = np.mean(data['treatment_converted'])
    
    # 3. 95% Confidence Interval для разницы долей
    diff_p = p2 - p1
    # Формула SE для пропорций
    se_p = np.sqrt((p1 * (1 - p1) / n1) + (p2 * (1 - p2) / n2))
    ci_p = [diff_p - 1.96 * se_p, diff_p + 1.96 * se_p]
    
    print(f"Conversion Diff: {diff_p:.4f}")
    print(f"Conversion 95% CI: [{ci_p[0]:.4f}, {ci_p[1]:.4f}]")




# Логика запуска (обнови свой блок if __name__ == "__main__")
if __name__ == "__main__":
    data = generate_data(seed=42)
    
    # 1. Описательная статистика
    describe_continuous("Control", data['control_duration'])
    describe_continuous("Treatment", data['treatment_duration'])
    describe_binary("Control", data['control_converted'])
    describe_binary("Treatment", data['treatment_converted'])
    
    # 2. Проверка условий
    check_assumptions(data['control_duration'], data['treatment_duration'])
    
    # 3. Тесты гипотез
    run_hypothesis_tests(data)
    
    # 4. Эффекты и интервалы
    calculate_effect_sizes(data)