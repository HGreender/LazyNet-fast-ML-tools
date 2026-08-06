import os
from tqdm import tqdm
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader

from fast_ml_tools.ml.trainer import Trainer
from fast_ml_tools.ml.metrics import get_segmentation_metrics
from fast_ml_tools.visualization import plot_epochs_data, show_segmentation
from fast_ml_tools.ml.augmentations import get_imagenet_encoder_augmentation
from fast_ml_tools.ml.utils import (preprocess_image_array_for_model,
                                    postprocess_prediction,
                                    get_device)
from fast_ml_tools.ml.models import (efficientnetb4_unet,
                                     efficientnetb4_unetpp)
from fast_ml_tools.ml.losses import (DiceBCELoss,
                                     FocalLoss,
                                     DiceFocalLoss,
                                     IoUFocalLoss)
from fast_ml_tools.ml.datasets import (create_train_val_datasets_from_multiple_dirs,
                                       MultiDirsDataset,
                                       create_weighted_sampler)

MODEL_FACTORIES = {
    'efficientnetb4_unet': efficientnetb4_unet,
    'efficientnetb4_unet++': efficientnetb4_unetpp,
}

LOSS_FACTORIES = {
    'focal': FocalLoss,
    'dice_bce': DiceBCELoss,
    'dice_focal': DiceFocalLoss,
    'iou_focal': IoUFocalLoss,
}


class LazyNet:
    ''' Класс для быстрого обучения моделей бинарной сегментации и инференса.
    Работает со списками папок с изображениями и масками.
    '''

    def __init__(
            self,
            model_name: str = None,
            model_path: str = None,
            model=None,
            loss_name: str = None,
            augmentation_fn: callable = None,
            train_img_dirs: list[str] = None,
            train_mask_dirs: list[str] = None,
            val_img_dirs: list[str] = None,
            val_mask_dirs: list[str] = None,
            mask_suffix: str | list[str] = "_mask",
            train_ratio: float = 0.8,
            train_val_seed: int = 42,
            early_stopping_threshold=10,
            epochs: int = 100,
            batch_size: int = 4,
            lr: float = 2.5e-4,
            lr_patience: int = 8,
            lr_factor: float = 0.2,
            weight_decay: float = 1e-4,
            verbose: bool = True,
            device_id: int = 0,
            num_workers: int = 0,
            classes: list = ['target_class'],
            metric_names: list = None,
            stratify_by_mask: bool = True,
            min_pixels_threshold: int = 10,
            positive_weight: float = 1.0
    ):
        self.device = get_device(device_id)
        self.verbose = verbose
        self.model_path = model_path
        self.mask_suffix = mask_suffix
        self.model = None

        # Параметры балансировки
        self.stratify_by_mask = stratify_by_mask
        self.min_pixels_threshold = min_pixels_threshold
        self.positive_weight = positive_weight

        if model is not None:
            self.model = model
        elif model_name is not None:
            if model_name not in MODEL_FACTORIES:
                raise ValueError(f"Неизвестная модель: {model_name}. Доступны: {self.available_models}")
            self.model = MODEL_FACTORIES[model_name](classes=classes)

        if self.model is not None:
            if model_path:
                self.load_weights(model_path)

        self.epochs = epochs
        self.metric_names = metric_names

        # Аугментации нужны всегда
        aug_fn = augmentation_fn or get_imagenet_encoder_augmentation
        self.train_augmentation = aug_fn(phase='train', size=(512, 512))
        self.val_augmentation = aug_fn(phase='valid', size=(512, 512))

        # 2. Датасеты и Лоадеры
        self.train_loader = None
        self.val_loader = None
        self.train_dataset = None
        self.val_dataset = None

        if train_img_dirs and train_mask_dirs:
            if len(train_img_dirs) != len(train_mask_dirs):
                raise ValueError("Количество папок с изображениями и масками для обучения должно совпадать.")

            if val_img_dirs is None or val_mask_dirs is None:
                self.train_dataset, self.val_dataset = create_train_val_datasets_from_multiple_dirs(
                    img_dirs=train_img_dirs,
                    mask_dirs=train_mask_dirs,
                    train_ratio=train_ratio,
                    seed=train_val_seed,
                    train_augmentation=self.train_augmentation,
                    val_augmentation=self.val_augmentation,
                    mask_suffix=self.mask_suffix,
                    stratify_by_mask=self.stratify_by_mask,
                    min_pixels_threshold=self.min_pixels_threshold
                )
            else:
                if len(val_img_dirs) != len(val_mask_dirs):
                    raise ValueError("Количество папок с изображениями и масками для валидации должно совпадать.")

                self.train_dataset = MultiDirsDataset(
                    img_dirs=train_img_dirs,
                    mask_dirs=train_mask_dirs,
                    augmentation=self.train_augmentation,
                    mask_suffix=self.mask_suffix
                )
                self.val_dataset = MultiDirsDataset(
                    img_dirs=val_img_dirs,
                    mask_dirs=val_mask_dirs,
                    augmentation=self.val_augmentation,
                    mask_suffix=self.mask_suffix
                )

            if self.train_dataset:
                # use_sampler = self.stratify_by_mask and isinstance(self.train_dataset, type(self.val_dataset))

                if isinstance(self.train_dataset, type(self.val_dataset)) and hasattr(self.train_dataset, 'samples'):
                    sampler = create_weighted_sampler(
                        self.train_dataset,
                        min_pixels_threshold=self.min_pixels_threshold,
                        positive_weight=self.positive_weight
                    )

                    self.train_loader = DataLoader(
                        self.train_dataset,
                        batch_size=batch_size,
                        sampler=sampler,  # Используем sampler вместо shuffle
                        num_workers=num_workers,
                        pin_memory=True
                    )
                else:
                    # Если используются разные классы датасетов или sampler не применим, используем обычный shuffle
                    self.train_loader = DataLoader(
                        self.train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True
                    )

            if self.val_dataset:
                self.val_loader = DataLoader(
                    self.val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True
                )

        # 3. Оптимизатор, Лосс и Трейнер
        self.optimizer = None
        self.loss_fn = None
        self.scheduler = None
        self.trainer = None

        # Трейнер создаем только если есть модель, данные и лосс
        if self.model is not None and self.train_loader and loss_name:
            self._init_optimizer_scheduler_trainer(
                lr=lr,
                weight_decay=weight_decay,
                lr_patience=lr_patience,
                lr_factor=lr_factor,
                loss_name=loss_name,
                early_stopping_threshold=early_stopping_threshold,
                metric_names=metric_names
            )

        self.train_logs = None

    def _init_optimizer_scheduler_trainer(
            self,
            lr: float,
            weight_decay: float,
            lr_patience: int,
            lr_factor: float,
            loss_name: str,
            early_stopping_threshold: int,
            metric_names: list = None
    ):
        """Вспомогательный метод для инициализации оптимизатора, scheduler и trainer"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        if loss_name not in LOSS_FACTORIES:
            raise ValueError(f"Неизвестная функция ошибки: {loss_name}. Доступны: {self.available_losses}")
        self.loss_fn = LOSS_FACTORIES[loss_name]()

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=lr_factor,
            patience=lr_patience
        )

        metrics = get_segmentation_metrics(metric_names)

        self.trainer = Trainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            early_stopping_threshold=early_stopping_threshold,
            metrics=metrics,
            scheduler=self.scheduler,
            device=self.device,
            verbose=self.verbose,
        )

    @property
    def available_models(self) -> list[str]:
        return list(MODEL_FACTORIES.keys())

    @property
    def available_losses(self) -> list[str]:
        return list(LOSS_FACTORIES.keys())

    def load_weights(self, path: str):
        """Загрузка весов модели"""
        if self.model is None:
            raise RuntimeError("Модель не инициализирована. Укажите model_name или передайте объект model.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл модели не найден: {path}")

        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        print(f'Веса загружены из {path}')

    def export_to_onnx(
            self,
            onnx_path: str = './lazy_data/models/model.onnx',
            input_shape: tuple = (1, 3, 512, 512),
            opset_version: int = 17,
            dynamic_axes: dict = None
    ):
        """
        Экспорт модели в ONNX формат.

        Args:
            onnx_path: Путь для сохранения ONNX модели
            input_shape: Форма входного тензора (batch, channels, height, width)
            opset_version: Версия ONNX opset
            dynamic_axes: Словарь для динамических осей (опционально)
        """
        if self.model is None:
            raise RuntimeError("Модель не инициализирована. Невозможно выполнить экспорт.")

        # Устанавливаем режим eval
        self.model.eval()

        # Создаем dummy input
        dummy_input = torch.randn(input_shape).to(self.device)

        # Определяем динамические оси (опционально)
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {},
                'output': {}
            }

        # Создаем директорию если нужно
        os.makedirs(os.path.dirname(onnx_path) or '.', exist_ok=True)

        # Экспортируем в ONNX с отключенным Dynamo
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            verbose=False,
            dynamo=False
        )

        print(f"Модель успешно экспортирована в ONNX: {onnx_path}")
        return onnx_path

    def fit(
            self,
            save_model_path: str = './lazy_data/models/best_model.pth',
            save_logs_path: str = './lazy_data/logs/training_logs.json',
            save_checkpoint_path: str = None,
            load_checkpoint_path: str = None
    ):
        if self.trainer is None:
            raise RuntimeError(
                "Невозможно запустить fit(): не инициализирован Trainer. "
                "Проверьте наличие model, train_img_dirs, train_mask_dirs и loss_name.")

        os.makedirs(os.path.dirname(save_model_path) or '.', exist_ok=True)
        os.makedirs(os.path.dirname(save_logs_path) or '.', exist_ok=True)

        self.train_logs = self.trainer.fit(
            epochs=self.epochs,
            save_path=save_model_path,
            log_path=save_logs_path,
            save_checkpoint_path=save_checkpoint_path,
            resume_from=load_checkpoint_path
        )

        self.model_path = save_model_path

    def fit_with_optuna(
            self,
            trial,
            save_dir: str = './optuna_trials',
            epochs: int = None,
            use_checkpoints: bool | None = None
    ):
        """
        Метод для обучения с поддержкой Optuna pruning.
        Возвращает лучший val_loss для минимизации.
        """
        if self.trainer is None:
            raise RuntimeError(
                "Невозможно запустить fit_with_optuna(): не инициализирован Trainer."
            )

        # Создаем директорию для trial
        trial_dir = os.path.join(save_dir, f'trial_{trial.number}')
        os.makedirs(trial_dir, exist_ok=True)

        save_model_path = os.path.join(trial_dir, 'best_model.pth')
        save_logs_path = os.path.join(trial_dir, 'logs.json')
        save_checkpoint_path = os.path.join(trial_dir, 'checkpoint.pth') if use_checkpoints else None

        # Обучаем с передачей trial для pruning
        self.train_logs = self.trainer.fit(
            epochs=epochs or self.epochs,
            save_path=save_model_path,
            log_path=save_logs_path,
            save_checkpoint_path=save_checkpoint_path,
            optuna_trial=trial
        )

        # --- ИЗВЛЕЧЕНИЕ ЛУЧШЕГО VAL LOSS ---
        best_val_loss = float('inf')

        if self.train_logs and hasattr(self.train_logs, 'val_losses'):
            # self.train_logs.val_losses - это список чисел (float)
            if self.train_logs.val_losses:
                best_val_loss = min(self.train_logs.val_losses)

        return best_val_loss

    def draw_logs(
            self,
            logs_path: str = None,
            show_metrics: bool = True,
            metric_names: list[str] = None
    ):
        if metric_names is None:
            metric_names = self.metric_names

        if self.train_logs is not None or logs_path:
            plot_epochs_data(
                self.train_logs,
                logs_path=logs_path,
                show_metrics=show_metrics,
                metric_names=metric_names
            )
        else:
            raise FileNotFoundError(f"Файл модели не найден: {logs_path}")

    def _predict_mask(
            self,
            image_rgb: np.ndarray,
            threshold: float = 0.5,
            return_probabilities: bool = False,
            class_idx: int = 0
    ):
        """
        Внутренний метод для получения маски и вероятностей из numpy array.

        Args:
            image_rgb: Изображение в формате RGB (H, W, C)
            threshold: Порог для бинаризации
            return_probabilities: Вернуть ли карту вероятностей
            class_idx: Индекс класса для многоклассовой модели

        Returns:
            tuple: (binary_mask, prob_map_resized, original_size)
        """
        if self.model is None:
            raise RuntimeError("Модель не инициализирована. Невозможно выполнить предсказание.")

        self.model.eval()

        # Препроцессинг изображения
        input_tensor, original_size = preprocess_image_array_for_model(
            image_rgb,
            self.val_augmentation,
            self.device
        )

        with torch.no_grad():
            output = self.model(input_tensor)

        # Получаем карту вероятностей до постпроцессинга
        num_channels = output.shape[1]
        if num_channels == 1:
            prob_map_raw = torch.sigmoid(output).squeeze().cpu().numpy()
        else:
            probs = torch.softmax(output, dim=1)
            prob_map_raw = probs[:, class_idx, :, :].squeeze().cpu().numpy()

        # Ресайзим карту вероятностей до оригинального размера
        prob_map_resized = cv2.resize(prob_map_raw, original_size, interpolation=cv2.INTER_LINEAR)

        # Получаем бинарную маску через существующую функцию
        binary_mask = postprocess_prediction(
            output,
            threshold,
            original_size=original_size,
            return_probabilities=False  # Мы сами работаем с prob_map_resized
        )

        if return_probabilities:
            return binary_mask, prob_map_resized, original_size
        return binary_mask, None, original_size

    def predict_single(
            self,
            image_path: str,
            threshold: float = 0.5,
            visualize: bool = True,
            save_concat_path: str = None,
            save_mask_path: str = None,
            return_probabilities: bool = False
    ):
        """Предсказание для одного изображения."""
        # Загружаем изображение
        image_cv = cv2.imread(image_path)
        if image_cv is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        binary_mask, prob_map, _ = self._predict_mask(
            image_rgb=image_rgb,
            threshold=threshold,
            return_probabilities=True
        )

        if save_mask_path:
            os.makedirs(os.path.dirname(save_mask_path) or '.', exist_ok=True)
            cv2.imwrite(save_mask_path, binary_mask * 255)
            print(f'Mask saved to {save_mask_path}')

        if visualize or save_concat_path:
            show_segmentation(
                image_rgb=image_rgb,
                binary_mask=binary_mask,
                save_path=save_concat_path,
                is_show=visualize
            )

        if return_probabilities:
            return binary_mask, prob_map
        return binary_mask

    def predict_image(
            self,
            image_rgb: np.ndarray,
            threshold: float = 0.5,
            return_probabilities: bool = False
    ):
        """
        Предсказание для numpy array изображения.

        Args:
            image_rgb: Изображение в формате RGB (H, W, C)
            threshold: Порог для бинаризации
            return_probabilities: Вернуть ли карту вероятностей

        Returns:
            binary_mask или (binary_mask, prob_map)
        """
        binary_mask, prob_map, _ = self._predict_mask(
            image_rgb=image_rgb,
            threshold=threshold,
            return_probabilities=True
        )

        if return_probabilities:
            return binary_mask, prob_map
        return binary_mask

    def visualize_heatmap_contours(
            self,
            image_path: str,
            min_threshold: float = 1e-5,
            save_path: str = None,
            is_show: bool = True,
            class_idx: int = 0
    ):
        """
        Визуализирует все контуры патологии и подписывает диапазон их активации.
        Текст автоматически корректируется, чтобы не выходить за границы изображения.
        """
        if self.model is None:
            raise RuntimeError("Модель не инициализирована.")

        # Загружаем изображение
        image_cv = cv2.imread(image_path)
        if image_cv is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение: {image_path}")
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        # Используем новый метод для получения данных
        _, prob_map, original_size = self._predict_mask(
            image_rgb=image_rgb,
            threshold=0.5,  # Порог здесь не важен, т.к. мы используем prob_map
            return_probabilities=True,
            class_idx=class_idx
        )

        h, w = original_size
        vis_img = image_rgb.copy()

        # Находим все связные компоненты выше min_threshold
        binary_mask = (prob_map >= min_threshold).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        print(f"Найдено {num_labels - 1} компонент(ов) выше порога {min_threshold}")

        # Цвета для различения компонентов
        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 0),
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35  # Уменьшенный шрифт
        thickness = 1

        for label in range(1, num_labels):  # Пропускаем фон (label=0)
            component_mask = (labels == label).astype(np.uint8)

            # Вычисляем min и max вероятности ВНУТРИ этого компонента
            component_probs = prob_map[component_mask > 0]

            if component_probs.size == 0:
                continue

            p_min = component_probs.min()
            p_max = component_probs.max()

            # Находим контур компонента
            contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            color = colors[(label - 1) % len(colors)]

            # Рисуем контур
            cv2.drawContours(vis_img, contours, -1, color, 2)

            # Формируем подпись
            text = f"{p_min:.5f}-{p_max:.3f}"

            # Определяем начальную позицию рядом с центроидом
            cx, cy = centroids[label]
            start_x = int(cx) + 5
            start_y = int(cy) - 5

            # --- ЛОГИКА ОГРАНИЧЕНИЯ ТЕКСТА В РАМКАХ ИЗОБРАЖЕНИЯ ---
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Корректировка по X (чтобы не уходило за правый край)
            if start_x + text_w > w:
                start_x = w - text_w - 2

            # Корректировка по Y (чтобы не уходило за нижний край)
            # text_h измеряется от базовой линии вверх, но нам нужно учесть и baseline
            if start_y + baseline > h:
                start_y = h - baseline - 2

            # Дополнительная проверка, чтобы текст не уходил за левый/верхний край при сильном сдвиге
            start_x = max(2, start_x)
            start_y = max(text_h + 2, start_y)

            text_pos = (start_x, start_y)

            # Рисуем тень (черная подложка) для читаемости
            cv2.putText(vis_img, text, text_pos, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            # Рисуем основной текст
            cv2.putText(vis_img, text, text_pos, font, font_scale, color, thickness, cv2.LINE_AA)

        if save_path:
            if not os.path.splitext(save_path)[1]:
                save_path += '.png'
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
            print(f"Визуализация сохранена в {save_path}")

        if is_show:
            try:
                cv2.imshow('Heatmap Contours Visualization', cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 12))
                plt.imshow(vis_img)
                plt.title(f'Contours & Activation Ranges (thr >= {min_threshold})')
                plt.axis('off')
                plt.show()

        return vis_img

    def predict_folder(
            self,
            input_dir: str,
            output_dir: str = './lazy_data/predictions/',
            threshold: float = 0.5,
            save_masks: bool = True,
            save_concat: bool = True,
            visualize_console: bool = False,
            mask_suffix: str = '_mask',
            viz_suffix: str = '_viz'
    ):
        """Пакетное предсказание для всех изображений в папке."""
        if self.model is None:
            raise RuntimeError("Модель не инициализирована. Невозможно выполнить предсказание.")

        self.model.eval()

        masks_dir = os.path.join(output_dir, 'masks') if save_masks else None
        concat_dir = os.path.join(output_dir, 'concat') if save_concat else None

        if masks_dir: os.makedirs(masks_dir, exist_ok=True)
        if concat_dir: os.makedirs(concat_dir, exist_ok=True)

        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

        if not files:
            print(f"В папке {input_dir} не найдено изображений.")
            return

        print(f"Начинаю обработку {len(files)} изображений...")

        for filename in tqdm(files, desc="Predicting"):
            img_path = os.path.join(input_dir, filename)
            name, ext = os.path.splitext(filename)

            try:
                self.predict_single(
                    img_path,
                    threshold=threshold,
                    visualize=visualize_console,
                    save_concat_path=os.path.join(concat_dir, f"{name}{viz_suffix}{ext}") if concat_dir else None,
                    save_mask_path=os.path.join(masks_dir, f"{name}{mask_suffix}.png") if masks_dir else None
                )
            except Exception as e:
                print(f"Ошибка при обработке {filename}: {e}")

        print(f"Готово! Результаты сохранены в {output_dir}")

    def visualize_folder_heatmaps(
            self,
            input_dir: str,
            output_dir: str = './lazy_data/heatmap_viz/',
            min_threshold: float = 1e-5,
            class_idx: int = 0,
            save_contours_only: bool = False
    ):
        """
        Пакетная визуализация тепловых карт и контуров для всех изображений в папке.

        Args:
            input_dir: Папка с исходными изображениями.
            output_dir: Папка для сохранения результатов.
            min_threshold: Минимальный порог чувствительности.
            class_idx: Индекс класса патологии.
            save_contours_only: Если True, сохраняет только изображение с разметкой (без оригинала рядом).
        """
        if self.model is None:
            raise RuntimeError("Модель не инициализирована.")

        self.model.eval()
        os.makedirs(output_dir, exist_ok=True)

        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_extensions)]

        if not files:
            print(f"В папке {input_dir} не найдено изображений.")
            return

        print(f"Начинаю пакетную обработку {len(files)} изображений...")

        for filename in tqdm(files, desc="Processing Heatmaps"):
            img_path = os.path.join(input_dir, filename)
            name, ext = os.path.splitext(filename)

            try:
                # Загружаем изображение
                image_cv = cv2.imread(img_path)
                if image_cv is None:
                    raise FileNotFoundError(f"Не удалось прочитать изображение: {img_path}")
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

                # Теперь используем единый метод для получения данных
                _, prob_map, original_size = self._predict_mask(
                    image_rgb=image_rgb,
                    threshold=0.5,
                    return_probabilities=True,
                    class_idx=class_idx
                )

                h, w = original_size
                vis_img = image_rgb.copy()

                binary_mask = (prob_map >= min_threshold).astype(np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

                colors = [
                    (0, 255, 0), (255, 0, 0),
                    (0, 255, 255), (255, 0, 255), (255, 255, 0)
                ]

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.35
                thickness = 1

                for label in range(1, num_labels):
                    component_mask = (labels == label).astype(np.uint8)
                    component_probs = prob_map[component_mask > 0]

                    if component_probs.size == 0: continue

                    p_min = component_probs.min()
                    p_max = component_probs.max()

                    contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    color = colors[(label - 1) % len(colors)]

                    cv2.drawContours(vis_img, contours, -1, color, 2)

                    text = f"{p_min:.5f}-{p_max:.3f}"
                    cx, cy = centroids[label]
                    start_x = int(cx) + 5
                    start_y = int(cy) - 5

                    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

                    if start_x + text_w > w: start_x = w - text_w - 2
                    if start_y + baseline > h: start_y = h - baseline - 2
                    start_x = max(2, start_x)
                    start_y = max(text_h + 2, start_y)

                    text_pos = (start_x, start_y)
                    cv2.putText(vis_img, text, text_pos, font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                    cv2.putText(vis_img, text, text_pos, font, font_scale, color, thickness, cv2.LINE_AA)

                # Сохранение результата
                save_name = f"{name}_viz{ext}"
                save_path = os.path.join(output_dir, save_name)
                cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

            except Exception as e:
                print(f"Ошибка при обработке {filename}: {e}")

        print(f"Готово! Результаты сохранены в {output_dir}")