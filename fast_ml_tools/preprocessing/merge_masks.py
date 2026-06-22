import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm


def merge_masks(
        input_folder: str,
        prefix1: str,
        prefix2: str,
        output_folder: str
) -> None:
    """
    Объединяет две маски для каждого изображения.

    Args:
        input_folder: Путь к папке с исходными масками
        prefix1: Первый префикс маски (например, 'mask_4')
        prefix2: Второй префикс маски (например, 'mask_6')
        output_folder: Путь к выходной папке
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Получаем все PNG файлы
    mask_files = list(input_path.glob("*.png"))

    if not mask_files:
        print(f"Не найдено PNG файлов в {input_folder}")
        return

    # Группируем маски по базовому имени изображения
    image_groups = {}

    for mask_file in mask_files:
        filename = mask_file.stem  # без расширения, например "or1113_mask_20"

        # Ищем "_mask_" и разделяем по нему
        mask_marker = "_mask_"
        if mask_marker not in filename:
            continue

        # Разделяем по первому вхождению "_mask_"
        parts = filename.split(mask_marker, 1)
        if len(parts) != 2:
            continue

        base_name = parts[0]  # "or1113"
        suffix = parts[1]  # "20"
        full_prefix = f"mask_{suffix}"  # "mask_20"

        if full_prefix not in [prefix1, prefix2]:
            continue

        if base_name not in image_groups:
            image_groups[base_name] = {}

        image_groups[base_name][full_prefix] = mask_file

    # Объединяем маски для каждого изображения
    merged_count = 0
    skipped_no_pair = 0

    for base_name, masks in tqdm(image_groups.items(), desc="Объединение масок"):
        if prefix1 not in masks or prefix2 not in masks:
            skipped_no_pair += 1
            continue

        try:
            # Загружаем маски
            mask1 = np.array(Image.open(masks[prefix1]))
            mask2 = np.array(Image.open(masks[prefix2]))

            # Проверяем одинаковые размеры
            if mask1.shape != mask2.shape:
                print(f"Пропущено {base_name}: разные размеры масок ({mask1.shape} vs {mask2.shape})")
                continue

            # Объединяем (логическое ИЛИ для бинарных масок)
            merged_mask = np.maximum(mask1, mask2)

            # Сохраняем результат
            output_file = output_path / f"{base_name}_mask.png"
            Image.fromarray(merged_mask).save(output_file)
            merged_count += 1

        except Exception as e:
            print(f"Ошибка обработки {base_name}: {e}")

    print(f"Объединено {merged_count} масок в {output_folder}")
    if skipped_no_pair > 0:
        print(f"Пропущено {skipped_no_pair} изображений (нет пары масок)")

# # Пример использования
# BASE_PATH = '/mnt/a100_data/datasets/global_localization_dataset/ФЛГ_РГ_ОГК/'
# merge_masks(
#     input_folder=BASE_PATH + "all_8700_v1/preprocessed/charless_min-max_no-interest/masks",
#     prefix1="mask_12",
#     prefix2="mask_26",
#     output_folder=BASE_PATH + "all_8700_v1/test"
# )
