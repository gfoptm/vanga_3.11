from datetime import timedelta
def get_interval_seconds(interval: str) -> int:
    """Преобразует строку интервала (например, '1h', '15m') в секунды."""
    if interval.endswith("m"):
        return int(interval[:-1]) * 60
    elif interval.endswith("h"):
        return int(interval[:-1]) * 3600
    elif interval.endswith("d"):
        return int(interval[:-1]) * 86400
    else:
        raise ValueError("Неподдерживаемый формат интервала")


def align_forecast_time(ft: int, interval: str) -> int:
    """
    Выравнивает время до ближайшей границы свечи.
    Пример: для интервала '1h': если ft соответствует 7:50, результат будет 8:00.
    """
    seconds = get_interval_seconds(interval)
    return ((ft + seconds - 1) // seconds) * seconds


def floor_forecast_time(ft: int, interval: str) -> int:
    """
    Округляет время вниз до начала интервала.
    Пример: для интервала '1h': если ft соответствует 13:50, результат будет 13:00.
    """
    seconds = get_interval_seconds(interval)
    return (ft // seconds) * seconds

def interval_to_timedelta(interval: str) -> timedelta:
    """
    Преобразует строку интервала ("1h", "15m", "1d") в объект timedelta.
    """
    if not isinstance(interval, str) or len(interval) < 2:
        raise ValueError(f"Неверный формат интервала: {interval}")

    unit = interval[-1].lower()
    try:
        value = int(interval[:-1])
    except ValueError:
        raise ValueError(f"Неверное числовое значение в интервале: {interval}")

    if unit == 'm':
        return timedelta(minutes=value)
    elif unit == 'h':
        return timedelta(hours=value)
    elif unit == 'd':
        return timedelta(days=value)
    else:
        raise ValueError(f"Неизвестная единица измерения в интервале: {unit}")