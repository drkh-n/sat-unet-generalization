import time as t

def format_time(seconds):
    """Форматирует время в формате часы:минуты:секунды"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:05.2f}"

def measure_time(func):
    """Декоратор для измерения времени выполнения функции"""
    def wrapper(*args, **kwargs):
        start_time = t.time()
        result = func(*args, **kwargs)
        end_time = t.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time {func.__name__}: {format_time(elapsed_time)}")
        return result
    return wrapper

# @measure_time
# def example_function():
#     i = 0
#     while i < 100:
#         i += 1
# example_function()

