#!/usr/bin/env python3
"""Объединение конкатов предсказаний моделей вертикально по именам файлов."""

from pathlib import Path
from PIL import Image


def merge_concatenations_by_name(input_folders: list[Path], output_folder: Path) -> None:
    """
    Объединяет конкаты из input_folders вертикально по совпадающим именам файлов.

    Args:
        input_folders: Список папок с входными конкатами
        output_folder: Папка для сохранения объединённых конкатов
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    # Собираем файлы из всех папок, группируем по имени
    files_by_name = {}
    for folder in input_folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"⚠️  Папка не найдена: {folder_path}")
            continue

        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
        for file in sorted(folder_path.iterdir()):
            if file.suffix.lower() in image_extensions and file.is_file():
                filename = file.name
                if filename not in files_by_name:
                    files_by_name[filename] = []
                files_by_name[filename].append((file, folder_path.name))

    if not files_by_name:
        print("❌ Не найдено файлов для объединения")
        return

    print(f"📁 Найдено {len(files_by_name)} уникальных имён файлов")

    # Обрабатываем каждое имя файла
    merged_count = 0
    for filename, file_list in sorted(files_by_name.items()):
        try:
            # Загружаем изображения
            images = []
            for file_path, folder_name in file_list:
                img = Image.open(file_path)
                images.append(img)

            if len(images) < 2:
                print(f"⚠️  Пропущен {filename}: только {len(images)} изображение")
                continue

            # Определяем общую ширину (берём максимальную)
            widths = [img.width for img in images]
            max_width = max(widths)

            # Масштабируем все к одинаковой ширине
            resized_images = []
            for img in images:
                if img.width != max_width:
                    ratio = max_width / img.width
                    new_height = int(img.height * ratio)
                    img_resized = img.resize((max_width, new_height), Image.LANCZOS)
                else:
                    img_resized = img
                resized_images.append(img_resized)

            # Вычисляем общую высоту
            total_height = sum(img.height for img in resized_images)

            # Создаём объединённое изображение
            merged = Image.new('RGB', (max_width, total_height), color='white')

            # Размещаем изображения вертикально
            y_offset = 0
            for img in resized_images:
                merged.paste(img, (0, y_offset))
                y_offset += img.height

            # Сохраняем результат
            output_path = output_folder / filename
            merged.save(output_path, quality=95)
            merged_count += 1
            print(f"✅ {filename}: {merged.width}x{merged.height}")

        except Exception as e:
            print(f"❌ Ошибка обработки {filename}: {e}")

    print(f"\n📊 Объединено {merged_count} файлов")


# # Пример использования
# INPUT_FOLDERS = [
#     Path("./unet_b4_1_pred/concat/"),
#     Path("./unet_b4_2_pred/concat/"),
#     Path("./unetpp_b4_1_pred/concat/"),
#     Path("./unetpp_b4_2_pred/concat/"),
#     Path("./unetpp_b4_3_pred/concat/"),
# ]
#
# OUTPUT_FOLDER = Path("./merged_results")
#
# merge_concatenations_by_name(INPUT_FOLDERS, OUTPUT_FOLDER)
