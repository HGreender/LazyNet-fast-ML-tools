import torch


def get_device(num: int = 0):
    """Получение устройства (CUDA/CPU)"""
    print(f"Доступно карт: {torch.cuda.device_count()}")
    if num >= torch.cuda.device_count() or num < 0:
        num = 0
        print("Выбрана дефолтная видеокарта")

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{num}")
    else:
        device = torch.device("cpu")
        if torch.version.cuda is None:
            print("ВНИМАНИЕ: torch собран БЕЗ CUDA (CPU-сборка), GPU не будет использован. "
                  "Переустановите torch с поддержкой CUDA, например: "
                  "pip install torch --index-url https://download.pytorch.org/whl/cu124 "
                  "(версию CUDA подбирайте под драйвер: nvidia-smi)")
        else:
            print(f"ВНИМАНИЕ: torch собран под CUDA {torch.version.cuda}, но GPU не найден. "
                  f"Проверьте драйвер: nvidia-smi (CUDA Version в шапке должна быть >= {torch.version.cuda})")

    print(f"Используемое устройство: {device}")
    return device
