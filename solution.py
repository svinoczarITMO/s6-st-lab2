import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import random

# Определение лингвистических переменных
speed_x = np.arange(0, 121, 1)  # Скорость (0-120 км/ч)
car_type_x = np.arange(1, 11, 1)  # Тип автомобиля (1-10)
traffic_x = np.arange(1, 11, 1)  # Трафик (1-10)
efficiency_x = np.arange(0, 101, 1)  # Эффективность (0-100%)

# Функции принадлежности для скорости
speed_low = fuzz.trimf(speed_x, [0, 0, 40])
speed_medium = fuzz.trimf(speed_x, [30, 60, 90])
speed_high = fuzz.trimf(speed_x, [70, 120, 120])

# Функции принадлежности для типа авто
car_compact = fuzz.trimf(car_type_x, [1, 1, 4])
car_sedan = fuzz.trimf(car_type_x, [3, 5, 7])
car_suv = fuzz.trimf(car_type_x, [6, 10, 10])

# Функции принадлежности для трафика
traffic_light = fuzz.trimf(traffic_x, [1, 1, 4])
traffic_moderate = fuzz.trimf(traffic_x, [3, 5, 7])
traffic_heavy = fuzz.trimf(traffic_x, [6, 10, 10])

# Функции принадлежности для эффективности
efficiency_low = fuzz.trimf(efficiency_x, [0, 0, 50])
efficiency_medium = fuzz.trimf(efficiency_x, [30, 50, 70])
efficiency_high = fuzz.trimf(efficiency_x, [50, 100, 100])

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

# Визуализация функций принадлежности
plot_variable(speed_x, [speed_low, speed_medium, speed_high],
            ['Низкая', 'Средняя', 'Высокая'],
            'Скорость движения', 'Скорость (км/ч)')

plot_variable(car_type_x, [car_compact, car_sedan, car_suv],
            ['Компактный', 'Седан', 'Внедорожник'],
            'Тип автомобиля', 'Тип (1-10)')

plot_variable(traffic_x, [traffic_light, traffic_moderate, traffic_heavy],
            ['Легкий', 'Умеренный', 'Тяжелый'],
            'Уровень трафика', 'Трафик (1-10)')

plot_variable(efficiency_x, [efficiency_low, efficiency_medium, efficiency_high],
            ['Низкая', 'Средняя', 'Высокая'],
            'Эффективность использования топлива', 'Эффективность (%)')

# Моделирование системы
for run in range(1, 6):
    # Генерация случайных входных значений
    speed_val = random.randint(0, 120)
    car_type_val = random.randint(1, 10)
    traffic_val = random.randint(1, 10)

    print(f"\nПрогон {run}:")
    print(f"Скорость: {speed_val} км/ч, Тип авто: {car_type_val}, Трафик: {traffic_val}")

    # Фазификация входных значений
    speed_low_val = fuzz.interp_membership(speed_x, speed_low, speed_val)
    speed_medium_val = fuzz.interp_membership(speed_x, speed_medium, speed_val)
    speed_high_val = fuzz.interp_membership(speed_x, speed_high, speed_val)

    car_compact_val = fuzz.interp_membership(car_type_x, car_compact, car_type_val)
    car_sedan_val = fuzz.interp_membership(car_type_x, car_sedan, car_type_val)
    car_suv_val = fuzz.interp_membership(car_type_x, car_suv, car_type_val)

    traffic_light_val = fuzz.interp_membership(traffic_x, traffic_light, traffic_val)
    traffic_moderate_val = fuzz.interp_membership(traffic_x, traffic_moderate, traffic_val)
    traffic_heavy_val = fuzz.interp_membership(traffic_x, traffic_heavy, traffic_val)

    # Применение нечетких правил
    # Правила для высокой эффективности
    rule1 = np.fmin(np.fmin(speed_low_val, car_compact_val), traffic_light_val)
    tip1 = np.fmin(rule1, efficiency_high)
    
    rule4 = np.fmin(speed_low_val, traffic_light_val)
    tip4 = np.fmin(rule4, efficiency_high)
    
    rule9 = np.fmin(car_sedan_val, speed_low_val)
    tip9 = np.fmin(rule9, efficiency_high)

    # Правила для средней эффективности
    rule3 = np.fmin(np.fmin(speed_medium_val, car_sedan_val), traffic_moderate_val)
    tip3 = np.fmin(rule3, efficiency_medium)
    
    rule7 = np.fmin(car_compact_val, traffic_moderate_val)
    tip7 = np.fmin(rule7, efficiency_medium)
    
    rule10 = np.fmin(traffic_light_val, car_sedan_val)
    tip10 = np.fmin(rule10, efficiency_medium)
    
    rule13 = np.fmin(speed_low_val, car_suv_val)
    tip13 = np.fmin(rule13, efficiency_medium)
    
    rule15 = np.fmin(speed_high_val, car_sedan_val)
    tip15 = np.fmin(rule15, efficiency_medium)

    # Правила для низкой эффективности
    rule2 = np.fmin(np.fmin(speed_high_val, car_suv_val), traffic_heavy_val)
    tip2 = np.fmin(rule2, efficiency_low)
    
    rule5 = np.fmin(car_suv_val, speed_high_val)
    tip5 = np.fmin(rule5, efficiency_low)
    
    rule6 = np.fmin(traffic_heavy_val, speed_medium_val)
    tip6 = np.fmin(rule6, efficiency_low)
    
    rule8 = np.fmin(speed_high_val, traffic_moderate_val)
    tip8 = np.fmin(rule8, efficiency_low)
    
    rule11 = np.fmin(speed_medium_val, traffic_heavy_val)
    tip11 = np.fmin(rule11, efficiency_low)
    
    rule12 = np.fmin(car_suv_val, traffic_moderate_val)
    tip12 = np.fmin(rule12, efficiency_low)
    
    rule14 = np.fmin(traffic_heavy_val, car_compact_val)
    tip14 = np.fmin(rule14, efficiency_low)

    # Агрегация и дефаззификация
    aggregated = np.fmax.reduce([tip1, tip2, tip3, tip4, tip5, tip6,
                               tip7, tip8, tip9, tip10, tip11, tip12,
                               tip13, tip14, tip15])
    
    if np.any(aggregated):
        result = fuzz.defuzz(efficiency_x, aggregated, 'centroid')
        print(f"Эффективность использования топлива: {result:.2f}%")
    else:
        print("Невозможно определить эффективность")

    # Визуализация результата
    plt.figure(figsize=(8, 4))
    plt.fill_between(efficiency_x, aggregated, facecolor='Gray', alpha=0.3)
    plt.plot(efficiency_x, efficiency_low, 'r--', label='Low')
    plt.plot(efficiency_x, efficiency_medium, 'g--', label='Medium')
    plt.plot(efficiency_x, efficiency_high, 'b--', label='High')
    plt.axvline(result, color='k', linestyle='--', label=f'Result: {result:.2f}%')
    plt.title(f'Вывод потребления (Прогон {run})')
    plt.legend()
    plt.grid(True)
    plt.show()