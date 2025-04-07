import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import random


# Тип маневров (1 - 10)
maneuver_x = np.arange(1, 11, 1)
maneuver_light = fuzz.trimf(maneuver_x, [1, 1, 4])
maneuver_moderate = fuzz.trimf(maneuver_x, [3, 5, 7])
maneuver_heavy = fuzz.trimf(maneuver_x, [6, 10, 10])

# Уровень подготовки водителя (1 - 10)
driver_x = np.arange(1, 11, 1)
driver_novice = fuzz.trimf(driver_x, [1, 1, 4])
driver_intermediate = fuzz.trimf(driver_x, [3, 5, 7])
driver_expert = fuzz.trimf(driver_x, [6, 10, 10])

# Состояние автомобиля (1 - 10)
condition_x = np.arange(1, 11, 1)
condition_poor = fuzz.trimf(condition_x, [1, 1, 4])
condition_fair = fuzz.trimf(condition_x, [3, 5, 7])
condition_good = fuzz.trimf(condition_x, [6, 10, 10])

# Качество топлива (1 - 10)
quality_x = np.arange(1, 11, 1)
quality_poor = fuzz.trimf(quality_x, [1, 1, 4])
quality_average = fuzz.trimf(quality_x, [3, 5, 7])
quality_good = fuzz.trimf(quality_x, [6, 10, 10])

# Скорость движения (0 - 120)
speed_x = np.arange(0, 121, 1)
speed_low = fuzz.trimf(speed_x, [0, 0, 40])
speed_medium = fuzz.trimf(speed_x, [30, 60, 90])
speed_high = fuzz.trimf(speed_x, [70, 120, 120])

# Траффик (1 - 10)
traffic_x = np.arange(1, 11, 1)
traffic_light = fuzz.trimf(traffic_x, [1, 1, 4])
traffic_moderate = fuzz.trimf(traffic_x, [3, 5, 7])
traffic_heavy = fuzz.trimf(traffic_x, [6, 10, 10])

# Тип автомобиля (1 - 10)
car_type_x = np.arange(1, 11, 1)
car_type_compact = fuzz.trimf(car_type_x, [1, 1, 4])
car_type_sedan = fuzz.trimf(car_type_x, [3, 5, 7])
car_type_suv = fuzz.trimf(car_type_x, [6, 10, 10])

# Потребление бензина (0 - 20 литров)
fuel_consumption_x = np.arange(0, 21, 1)
fuel_consumption_low = fuzz.trimf(fuel_consumption_x, [0, 0, 10])
fuel_consumption_medium = fuzz.trimf(fuel_consumption_x, [5, 10, 15])
fuel_consumption_high = fuzz.trimf(fuel_consumption_x, [10, 20, 20])


def plot_variable(x, mf_list, labels, title, xlabel):
    plt.figure(figsize=(8, 5))
    for mf, label in zip(mf_list, labels):
        plt.plot(x, mf, label=label)
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel('Степень принадлежности')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


plot_variable(maneuver_x, [maneuver_light, maneuver_moderate, maneuver_heavy],
              ['Light (легкие)', 'Moderate (умеренные)', 'Heavy (тяжелые)'],
              'Тип маневров', 'Тип маневров (1-10)')

plot_variable(driver_x, [driver_novice, driver_intermediate, driver_expert],
              ['Novice (новичок)', 'Intermediate (средний)', 'Expert (эксперт)'],
              'Уровень подготовки водителя', 'Уровень (1-10)')

plot_variable(condition_x, [condition_poor, condition_fair, condition_good],
              ['Poor (плохое)', 'Fair (среднее)', 'Good (хорошее)'],
              'Состояние автомобиля', 'Состояние (1-10)')

plot_variable(quality_x, [quality_poor, quality_average, quality_good],
              ['Poor (низкое)', 'Average (среднее)', 'Good (высокое)'],
              'Качество топлива', 'Качество топлива (1-10)')

plot_variable(speed_x, [speed_low, speed_medium, speed_high],
              ['Low (низкая)', 'Medium (средняя)', 'High (высокая)'],
              'Скорость', 'Скорость (км/ч)')

plot_variable(traffic_x, [traffic_light, traffic_moderate, traffic_heavy],
              ['Light (легкий)', 'Moderate (умеренный)', 'Heavy (тяжелый)'],
              'Траффик', 'Траффик (1-10)')

plot_variable(car_type_x, [car_type_compact, car_type_sedan, car_type_suv],
              ['Compact (компактный)', 'Sedan (седан)', 'SUV (внедорожник)'],
              'Тип автомобиля', 'Тип автомобиля (1-10)')


for run in range(1, 6):
    maneuver_val = random.randint(1, 10)
    driver_skill_val = random.randint(1, 10)
    condition_val = random.randint(1, 10)
    fuel_quality_val = random.randint(1, 10)
    speed_val = random.randint(0, 120)
    traffic_val = random.randint(1, 10)
    car_type_val = random.randint(1, 10)

    print(f"\nВходные данные:")
    print(f"Тип маневров: {maneuver_val}, Уровень подготовки водителя: {driver_skill_val}, "
          f"Состояние автомобиля: {condition_val}, Качество топлива: {fuel_quality_val}, "
          f"Скорость: {speed_val} км/ч, Трафик: {traffic_val}, Тип автомобиля: {car_type_val}")

    maneuver_light_val = fuzz.interp_membership(maneuver_x, maneuver_light, maneuver_val)
    maneuver_moderate_val = fuzz.interp_membership(maneuver_x, maneuver_moderate, maneuver_val)
    maneuver_heavy_val = fuzz.interp_membership(maneuver_x, maneuver_heavy, maneuver_val)

    driver_novice_val = fuzz.interp_membership(driver_x, driver_novice, driver_skill_val)
    driver_intermediate_val = fuzz.interp_membership(driver_x, driver_intermediate, driver_skill_val)
    driver_expert_val = fuzz.interp_membership(driver_x, driver_expert, driver_skill_val)

    condition_poor_val = fuzz.interp_membership(condition_x, condition_poor, condition_val)
    condition_fair_val = fuzz.interp_membership(condition_x, condition_fair, condition_val)
    condition_good_val = fuzz.interp_membership(condition_x, condition_good, condition_val)

    quality_poor_val = fuzz.interp_membership(quality_x, quality_poor, fuel_quality_val)
    quality_average_val = fuzz.interp_membership(quality_x, quality_average, fuel_quality_val)
    quality_good_val = fuzz.interp_membership(quality_x, quality_good, fuel_quality_val)

    speed_low_val = fuzz.interp_membership(speed_x, speed_low, speed_val)
    speed_medium_val = fuzz.interp_membership(speed_x, speed_medium, speed_val)
    speed_high_val = fuzz.interp_membership(speed_x, speed_high, speed_val)

    traffic_light_val = fuzz.interp_membership(traffic_x, traffic_light, traffic_val)
    traffic_moderate_val = fuzz.interp_membership(traffic_x, traffic_moderate, traffic_val)
    traffic_heavy_val = fuzz.interp_membership(traffic_x, traffic_heavy, traffic_val)

    car_type_compact_val = fuzz.interp_membership(car_type_x, car_type_compact, car_type_val)
    car_type_sedan_val = fuzz.interp_membership(car_type_x, car_type_sedan, car_type_val)
    car_type_suv_val = fuzz.interp_membership(car_type_x, car_type_suv, car_type_val)

    
    # Правило 1
    rule1 = np.fmin(np.fmin(maneuver_light_val, driver_novice_val), condition_good_val)
    tip1 = np.fmin(rule1, fuel_consumption_low)

    # Правило 2
    rule2 = np.fmin(np.fmin(maneuver_heavy_val, driver_expert_val), quality_good_val)
    tip2 = np.fmin(rule2, fuel_consumption_medium)

    # Правило 3
    rule3 = np.fmin(condition_poor_val, quality_poor_val)
    tip3 = np.fmin(rule3, fuel_consumption_high)

    # Правило 4
    rule4 = np.fmin(speed_high_val, traffic_heavy_val)
    tip4 = np.fmin(rule4, fuel_consumption_high)

    # Правило 5
    rule5 = np.fmin(speed_low_val, car_type_compact_val)
    tip5 = np.fmin(rule5, fuel_consumption_low)

    # Правило 6
    rule6 = np.fmin(driver_intermediate_val, quality_average_val)
    tip6 = np.fmin(rule6, fuel_consumption_medium)

    # Правило 7
    rule7 = np.fmin(maneuver_moderate_val, traffic_moderate_val)
    tip7 = np.fmin(rule7, fuel_consumption_medium)

    # Правило 8
    rule8 = np.fmin(car_type_suv_val, speed_high_val)
    tip8 = np.fmin(rule8, fuel_consumption_high)

    # Правило 9
    rule9 = np.fmin(driver_novice_val, traffic_heavy_val)
    tip9 = np.fmin(rule9, fuel_consumption_high)

    # Правило 10
    rule10 = np.fmin(np.fmin(condition_good_val, quality_good_val), speed_medium_val)
    tip10 = np.fmin(rule10, fuel_consumption_low)

    # --- Агрегация ---
    aggregated = np.fmax.reduce([tip1, tip2, tip3, tip4, tip5, tip6, tip7, tip8, tip9, tip10])

    # --- Дефаззификация с проверкой на пустую область ---
    if np.any(aggregated):
        result = fuzz.defuzz(fuel_consumption_x, aggregated, 'centroid')
        print(f"Рекомендуемое потребление бензина: {result:.2f} литров")
    else:
        print("Ошибка: область членства пуста, не удалось провести дефаззификацию.")

    # --- Визуализация ---
    plt.figure(figsize=(8, 4))
    plt.fill_between(fuel_consumption_x, np.zeros_like(fuel_consumption_x), aggregated, facecolor='gray', alpha=0.7)
    plt.plot(fuel_consumption_x, fuel_consumption_low, 'b--', label='Low')
    plt.plot(fuel_consumption_x, fuel_consumption_medium, 'g--', label='Medium')
    plt.plot(fuel_consumption_x, fuel_consumption_high, 'r--', label='High')
    plt.axvline(result, color='black', linestyle=':', label=f'Result: {result:.2f} литров')
    plt.title(f'Вывод потребления (Прогон {run})')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
