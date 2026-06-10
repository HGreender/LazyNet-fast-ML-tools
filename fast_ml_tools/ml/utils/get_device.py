import torch


def get_device(num: int = 0):
    """Получение устройства (CUDA/CPU)"""
    print(f"Доступно карт: {torch.cuda.device_count()}")
    if num >= torch.cuda.device_count() or num < 0:
        num = 0
        print("Выбрана дефолтная видеокарта")

    device = torch.device(f"cuda:{num}" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")
    return device
