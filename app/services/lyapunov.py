import numpy as np


def compute_lyapunov_exponent(time_series: np.ndarray, m: int = 3, tau: int = 1,
                              min_separation: int = None) -> float:
    """
    Усовершенствованное вычисление наибольшего показателя Ляпунова с использованием
    реконструкции фазового пространства по методике Такенса и алгоритма ближайших соседей (метод Розенштейна).

    Аргументы:
      time_series: одномерный numpy-массив (например, массив закрывающих цен)
      m: размерность встраивания (например, 3-5)
      tau: временной сдвиг для построения задержанных векторов
      min_separation: минимальное временное расстояние между индексами, если None – используется tau

    Возвращает:
      Оценку наибольшего показателя Ляпунова (λ) в единицах обратных шагов времени.
      Если данных недостаточно, возвращает 0.0.
    """
    N = len(time_series)
    if min_separation is None:
        min_separation = tau  # минимальная разница индексов, чтобы не брать соседей, близких по времени

    # Количество векторов задержек
    M = N - (m - 1) * tau
    if M < 2:
        return 0.0

    # Реконструкция фазового пространства: формируем матрицу, где каждая строка — вектор размерности m
    embedded = np.empty((M, m))
    for i in range(M):
        embedded[i] = time_series[i: i + m * tau: tau]

    # Для каждой точки находим индекс ближайшего соседа с учетом временного разрыва
    nearest_index = np.zeros(M, dtype=int)
    for i in range(M):
        min_dist = np.inf
        nn_index = -1
        for j in range(M):
            if abs(i - j) < min_separation:
                continue
            dist = np.linalg.norm(embedded[i] - embedded[j])
            if dist < min_dist:
                min_dist = dist
                nn_index = j
        nearest_index[i] = nn_index

    # Определим максимальное количество шагов времени для оценки дивергенции
    max_t = int(np.floor(M / 10))
    if max_t < 2:
        return 0.0

    # Массивы для накопления логарифмов расстояний
    divergence = np.zeros(max_t)
    counts = np.zeros(max_t)

    # Для каждой пары (i, nearest_index[i]), считаем логарифмическое расстояние через k шагов
    for i in range(M - max_t):
        j = nearest_index[i]
        if j < 0 or j >= M - max_t:
            continue
        for k in range(max_t):
            # Формируем k-шаговые векторы
            diff = embedded[i + k] - embedded[j + k]
            d = np.linalg.norm(diff)
            # Для стабильности избегаем логарифма 0
            if d > 1e-8:
                divergence[k] += np.log(d)
                counts[k] += 1

    # Усредняем логарифмическую дивергенцию для каждого k
    valid = counts > 0
    avg_divergence = divergence[valid] / counts[valid]
    times = np.arange(len(avg_divergence))

    # Линейная аппроксимация: наклон логарифмической зависимости является оценкой λ
    if len(times) < 2:
        return 0.0

    slope, intercept = np.polyfit(times, avg_divergence, 1)
    return round(slope, 4)
