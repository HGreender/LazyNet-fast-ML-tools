import json
import os
import numpy as np
from PIL import Image, ImageDraw

# Для работы с DICOM (если оригиналы в .dcm)
try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def get_image_size_from_json(item):
    """Пытается получить размер из метаданных JSON"""
    try:
        if 'width' in item and 'height' in item:
            return (item['width'], item['height'])
        elif 'size' in item['image']:
            return (item['image']['size'][1], item['image']['size'][0])
        else:
            return None
    except Exception:
        return None


def get_image_size_from_file(item_id, base_dir, is_dicom):
    """Пытается получить размер, открыв оригинальный файл"""
    # Пробуем разные расширения
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'] if not is_dicom else ['.dcm']

    for ext in extensions:
        # Для default.json ID обычно совпадают с именем файла
        # Для default2.json (DICOM UID) имя файла может отличаться, нужна осторожность
        file_path = os.path.join(base_dir, f"{item_id}{ext}")

        if os.path.exists(file_path):
            try:
                if is_dicom and HAS_PYDICOM:
                    ds = pydicom.dcmread(file_path)
                    return (ds.Columns, ds.Rows)
                else:
                    with Image.open(file_path) as img:
                        return img.size
            except Exception as e:
                print(f"Ошибка чтения файла {file_path}: {e}")
        # Если точное совпадение не найдено, можно попробовать поискать файл, начинающийся с ID
        # (актуально для DICOM, где UID длинный)
        if not ext:
            # Поиск файла по началу имени (медленно, но надежно для DICOM)
            for fname in os.listdir(base_dir):
                if fname.startswith(item_id.split('.')[0]):  # Грубое совпадение
                    fpath = os.path.join(base_dir, fname)
                    # ... логика чтения ...
                    pass
    return None


def create_mask_from_polygon(polygon_points, img_size):
    mask = Image.new('L', img_size, 0)
    pts = list(zip(polygon_points[0::2], polygon_points[1::2]))
    if len(pts) > 2:
        ImageDraw.Draw(mask).polygon(pts, outline=1, fill=1)
    return np.array(mask)


def process_json_to_mask(
        json_path: str,
        originals_dir: str,
        output_dir: str,
        label_id: int,
        with_postfix: bool = False,
        is_dicom: bool = False
):
    '''
    :param json_path: path of json annotations file
    :param output_dir: path of output directory for masks
    :param originals_dir: path of original images directory
    :param label_id: pathology ID in the annotation
    :param with_postfix: whether to add postfix to the output mask
    :param is_dicom: if True, parse DICOM
    '''
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    items = data.get('items', [])
    print(f"Найдено элементов: {len(items)}")

    for i, item in enumerate(items):
        img_id = str(item.get('id', 'unknown'))
        # Очистка ID для имени файла (особенно важно для DICOM UID)
        safe_id = img_id.replace('/', '_').replace('\\', '_').replace(':', '_').replace('.', '_')

        # 1. Пытаемся получить размер из JSON
        size = get_image_size_from_json(item)

        # 2. Если нет в JSON, пытаемся прочитать из файла
        if size is None and originals_dir:
            size = get_image_size_from_file(img_id, originals_dir, is_dicom)

        if size is None:
            print(f"⚠️ Предупреждение: Не удалось определить размер для {img_id}. Пропуск.")
            continue

        width, height = size
        annotations = item.get('annotations', [])
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        has_annotation = False
        for ann in annotations:
            if ann.get('label_id') == label_id and ann.get('type') == 'polygon':
                points = ann.get('points', [])
                if points:
                    poly_mask = create_mask_from_polygon(points, (width, height))
                    combined_mask = np.maximum(combined_mask, poly_mask)
                    has_annotation = True

        if has_annotation:
            mask_img = Image.fromarray(combined_mask * 255)
            if with_postfix:
                save_path = os.path.join(output_dir, f"{img_id}_mask.png")
            else:
                save_path = os.path.join(output_dir, f"{img_id}.png")
            mask_img.save(save_path)
            if i % 10 == 0:
                print(f"Обработано: {i}/{len(items)} (Размер: {width}x{height})")
    print("Обработка завершена!")