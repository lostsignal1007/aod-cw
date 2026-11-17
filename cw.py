import numpy as np
import pandas as pd
import pyvista as pv
from scipy.optimize import minimize
from scipy.stats import norm, weibull_min
from scipy.interpolate import griddata
import time

t_start_total = time.perf_counter()
t0 = time.perf_counter()

# 1. Читаем строго лист "DATA" и пропускаем пустую первую строку (заголовки)
# Мы берем диапазон колонок, где гарантированно идут числовые матрицы (например, с 5 по 8 столбцы)
df = pd.read_excel("f2.xlsx", sheet_name="DATA", header=None)

# 2. Переводим запятые в точки (для экспоненциального формата вроде 2,4E-01 -> 2.4E-01)
df_numeric = df.replace(',', '.', regex=True).apply(pd.to_numeric, errors='coerce')

# Из скриншота видно, что полезные числовые данные без текста идут в столбцах с индексами 5, 6, 7
# Вытаскиваем эти три столбца и удаляем из них строки, где есть пустые ячейки (NaN)
clean_data = df_numeric[[5, 6, 7]].dropna().to_numpy(dtype=np.float64)

# 3. Присваиваем роли колонкам:
# Столбец 5 -> условная координата X (шаг сканирования)
# Столбец 6 -> условная координата Y (смещение профиля)
# Столбец 7 -> физическая высота рельефа Z (наши "ожоги" и шероховатость)
x_raw = clean_data[:, 0]
y_raw = clean_data[:, 1]
z_raw = clean_data[:, 2]

print(f"--- [УСПЕХ] Данные с листа DATA успешно импортированы ---")
print(f"Количество считанных 3D-точек: {len(z_raw)}")
print(f"Диапазон высот Z (микрорельеф): от {z_raw.min():.4f} до {z_raw.max():.4f}\n")

# 4. Строим регулярную расчетную 3D-сетку 50х50 для PyVista
x_grid = np.linspace(x_raw.min(), x_raw.max(), 50)
y_grid = np.linspace(y_raw.min(), y_raw.max(), 50)
X, Y = np.meshgrid(x_grid, y_grid)

# Интерполируем методом ближайших соседей (nearest), чтобы гарантировать отсутствие геометрических ошибок
Z_raw = griddata((x_raw, y_raw), z_raw, (X, Y), method='nearest')

# Для совместимости со старым кодом: вытягиваем данные в плоский массив для МНК
z_f = Z_raw.flatten()
x_f = X.flatten()
y_f = Y.flatten()

t_stage1 = (time.perf_counter() - t0) * 1000  

# ==============================================================================
# ЭТАП 2: АЛГОРИТМ МНК (Устранение системного наклона / Детрендинг)
# ==============================================================================

t0 = time.perf_counter()

# Разворачиваем матрицы в одномерные векторы для линейной алгебры
x_f = X.flatten()
y_f = Y.flatten()
z_f = Z_raw.flatten()

# Матрица планов для уравнения плоскости: Z = a*X + b*Y + c
A = np.c_[x_f, y_f, np.ones_like(x_f)]

# Находим коэффициенты [a, b, c] с помощью метода наименьших квадратов (МНК)
coefficients, _, _, _ = np.linalg.lstsq(A, z_f, rcond=None)
a, b, c = coefficients

# Вычисляем плоскость тренда
Z_trend = (a * X + b * Y + c)

# Очищаем исходные данные от наклона: теперь плоскость лежит в Z = 0
Z_clean = Z_raw - Z_trend

print("--- [МНК] Коррекция геометрии выполнена ---")
print(f"Уравнение плоскости наклона: Z = {a:.4f}*X + {b:.4f}*Y + {c:.4f}\n")

t_stage2 = (time.perf_counter() - t0) * 1000

# ==============================================================================
# ЭТАП 3: ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ (Автоматический поиск границ ожога)
# ==============================================================================

t0 = time.perf_counter()

# Берем края матрицы (например, первые 15 строк), где гарантированно нет ожога
reference_zone = Z_clean[:15, :15]

# Считаем параметры нормального распределения чистой поверхности
mu_noise = np.mean(reference_zone)
sigma_noise = np.std(reference_zone)

# Строим двусторонний доверительный интервал для уровня значимости альфа = 0.01 (99% надежности)
# Критическое значение Z-score для 99% составляет примерно 2.576
confidence_level = 0.99
z_score = norm.ppf(1 - (1 - confidence_level) / 2)

lower_bound = mu_noise - z_score * sigma_noise
upper_bound = mu_noise + z_score * sigma_noise

# Маска дефектов: все точки, вышедшие за границы доверительного интервала шума прибора
# Код автоматически маркирует структуру: 0 - норма, 1 - ожог/дефект
Defect_Mask = (Z_clean < lower_bound) | (Z_clean > upper_bound)

print("--- [Статистика] Сегментация дефектов завершена ---")
print(f"Доверительный интервал чистой поверхности: [{lower_bound:.4f}; {upper_bound:.4f}]")
damaged_percentage = (np.sum(Defect_Mask) / Defect_Mask.size) * 100
print(f"Автоматически определенная площадь повреждения: {damaged_percentage:.2f}%\n")

t_stage3 = (time.perf_counter() - t0) * 1000

# ==============================================================================
# ЭТАП 4: МЕТОД МАКСИМАЛЬНОГО ПРАВДОПОДОБИЯ (ММП) (Анализ природы разрушения)
# ==============================================================================

t0 = time.perf_counter()

# Вытаскиваем значения глубин Z исключительно из поврежденной зоны
damaged_depths = Z_clean[Defect_Mask]

# Так как в кратере значения отрицательные (вмятины), инвертируем их для анализа распределений
positive_defects = np.abs(damaged_depths)

# Отрицательный логарифм функции правдоподобия для распределения Вейбулла
def weibull_neg_log_likelihood(params, data):
    shape, scale = params
    if shape <= 0 or scale <= 0:
        return 1e10  # Возвращаем штраф для невалидных параметров
    # По формуле плотности распределения Вейбулла
    return -np.sum(weibull_min.logpdf(data, c=shape, scale=scale))

# Начальное приближение для численной оптимизации ММП
initial_guess = [1.0, np.mean(positive_defects)]

# Запуск максимизации правдоподобия через минимизацию scipy
mle_result = minimize(weibull_neg_log_likelihood, initial_guess, args=(positive_defects,), method='Nelder-Mead')
mle_shape, mle_scale = mle_result.x

print("--- [ММП] Оценка характера микрорельефа выполнена ---")
print(f"Параметр формы Вейбулла (Shape): {mle_shape:.4f}")
print(f"Параметр масштаба Вейбулла (Scale): {mle_scale:.4f}")

# Экспертный вывод программы на основе вычисленного распределения
if mle_shape < 1.8:
    verdict = "Критическая эрозия структуры, глубокое выгорание, высокая шероховатость наплывов."
else:
    verdict = "Плавное термическое оплавление поверхности, структура относительно стабильна."
print(f"Инженерное заключение: {verdict}\n")

t_stage4 = (time.perf_counter() - t0) * 1000
t_total = (time.perf_counter() - t_start_total) * 1000

print("\n=== ПРОФИЛИРОВАНИЕ ВРЕМЕНИ ВЫПОЛНЕНИЯ (мс) ===")
print(f"Этап 1 (Импорт и сетка): {t_stage1:.2f} мс ({t_stage1/t_total*100:.1f}%)")
print(f"Этап 2 (Детрендинг МНК): {t_stage2:.2f} мс ({t_stage2/t_total*100:.1f}%)")
print(f"Этап 3 (Сегментация):    {t_stage3:.2f} мс ({t_stage3/t_total*100:.1f}%)")
print(f"Этап 4 (Оптимизация ММП):{t_stage4:.2f} мс ({t_stage4/t_total*100:.1f}%)")
print(f"ИТОГО:                   {t_total:.2f} мс (100%)")

# ==============================================================================
# ЭТАП 5: РАЗДЕЛЬНАЯ ТРЁХМЕРНАЯ ВИЗУАЛИЗАЦИЯ (Идеальные геометрические близнецы)
# ==============================================================================


print("--- Запуск раздельной визуализации PyVista ---")

# 1. ПОДГОТОВКА СЕТОК (Оригинал и Анализ)
grid_honest_raw = pv.StructuredGrid(X, Y, Z_raw)
grid_trend = pv.StructuredGrid(X, Y, Z_trend)

# Для выровненных моделей используем абсолютно одинаковую базовую сетку высот Z_clean!
grid_clean_a = pv.StructuredGrid(X, Y, Z_clean)
grid_clean_a.point_data["Высота"] = Z_clean.flatten()

grid_clean_b = pv.StructuredGrid(X, Y, Z_clean)
grid_clean_b.point_data["Ожоги"] = Defect_Mask.flatten().astype(float)

# --- НАСТРОЙКИ МАСШТАБА ---
Z_SCALE = 3.0 

# Единые параметры сетки без жесткой фиксации bounds (PyVista сама выровняет их по одинаковой сетке Z_clean)
grid_settings = dict(
    xtitle="X (Шаг)", 
    ytitle="Y (Смещение)", 
    ztitle="Z (Высота)",
    font_size=8,              
    fmt="%.2f",               
    location="outer",
    all_edges=False,         
    n_xlabels=3,              
    n_ylabels=3,              
    n_zlabels=3
)

# --- ОКНО 1: ИСХОДНЫЕ ДАННЫЕ И ТРЕНД МНК ---
print("\n[ОКНО 1] Открывается исходная деталь с плоскостью тренда...")
plotter_raw = pv.Plotter(window_size=(1000, 700), title="1. Оригинальные данные + Тренд МНК")
plotter_raw.add_text("Входные данные: Физический наклон + Изгиб детали", font_size=12)

plotter_raw.set_scale(zscale=Z_SCALE)
plotter_raw.add_mesh(grid_honest_raw, color="gray", show_edges=True, label="Сырые данные")
plotter_raw.add_mesh(grid_trend, color="orange", opacity=0.35, lighting=False, label="Плоскость МНК")

plotter_raw.show_grid(**grid_settings)
plotter_raw.add_legend(bcolor=None, face=None)
plotter_raw.view_isometric()
plotter_raw.show(screenshot="original_data_plot.png")


# --- ОКНО 2: СТАТИСТИЧЕСКИЙ АНАЛИЗ (ИДЕНТИЧНАЯ ГЕОМЕТРИЯ) ---
print("\n[ОКНО 2] Открывается панель статистического анализа...")
plotter_analysis = pv.Plotter(shape=(1, 2), window_size=(1400, 650), title="2. Результаты статистического анализа")

# Задаем единый масштаб отображения для обоих сабплотов
plotter_analysis.set_scale(zscale=Z_SCALE)

# Сабплот А: Физический рельеф без системного наклона
plotter_analysis.subplot(0, 0)
plotter_analysis.add_text("А. Выровненный рельеф (Устранен наклон МНК)", font_size=11)
plotter_analysis.add_mesh(
    grid_clean_a, 
    scalars="Высота", 
    cmap="jet", 
    lighting=True,
    show_edges=False
)
plotter_analysis.add_scalar_bar(title="Высота Z", label_font_size=8, title_font_size=9)
plotter_analysis.show_grid(**grid_settings)


# Сабплот Б: Локализация ожогов по Доверительному Интервалу
plotter_analysis.subplot(0, 1)
plotter_analysis.add_text("Б. Статистическая карта прижогов (99% CI)", font_size=11)

# ИСПРАВЛЕНО: Отображаем ту же самую объемную сетку grid_clean_b, 
# но перекрашиваем её по маске ожогов. Теперь геометрия, длина и изгиб станут 1-в-1 как слева!
plotter_analysis.add_mesh(
    grid_clean_b, 
    scalars="Ожоги", 
    cmap="coolwarm", 
    n_colors=2, 
    lighting=True,
    show_edges=False
)
plotter_analysis.add_scalar_bar(title="0-Норма, 1-Ожог", label_font_size=8, title_font_size=9)
plotter_analysis.show_grid(**grid_settings)


# Синхронизация и вывод на экран
plotter_analysis.link_views()
plotter_analysis.view_isometric()
plotter_analysis.show(screenshot="statistical_analysis_plot.png")

print("\n--- Визуализация и экспорт скриншотов успешно завершены! ---")
