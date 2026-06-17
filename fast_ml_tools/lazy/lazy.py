import os

from tqdm import tqdm
import cv2
import torch
from torch.utils.data import DataLoader

from fast_ml_tools.ml.datasets import create_train_val_datasets_from_multiple_dirs, MultiDirsDataset
from fast_ml_tools.ml.models import efficientnetb4_unet, efficientnetb4_unetpp
from fast_ml_tools.ml.trainer import Trainer
from fast_ml_tools.ml.augmentations import get_imagenet_encoder_augmentation
from fast_ml_tools.ml.losses import DiceBCELoss, FocalLoss, DiceFocalLoss, IoUFocalLoss
from fast_ml_tools.visualization import plot_epochs_data, show_segmentation
from fast_ml_tools.ml.metrics import get_segmentation_metrics
from fast_ml_tools.ml.utils import preprocess_image_for_model, postprocess_prediction, get_device

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
            model_name: str,
            model_path: str = None,
            loss_name: str = None,
            train_img_dirs: list[str] = None,
            train_mask_dirs: list[str] = None,
            val_img_dirs: list[str] = None,
            val_mask_dirs: list[str] = None,
            mask_suffix: str = "",
            train_ratio: float = 0.8,
            train_val_seed: int = 42,
            early_stopping_threshold=10,
            epochs: int = 100,
            batch_size: int = 1,
            lr: float = 1e-4,
            lr_patience: int = 8,
            lr_factor: float = 0.1,
            verbose: bool = True,
            device_id: int = 0,
            num_workers: int = 0,
            classes: list = ['target_class'],
            metric_names: list = None
    ):
        self.device = get_device(device_id)
        self.verbose = verbose
        self.model_path = model_path
        self.mask_suffix = mask_suffix

        # 1. Инициализация модели
        if model_name is None and model_path is not None:
            raise ValueError("Необходимо указать model_name.")
        if model_name not in MODEL_FACTORIES:
            raise ValueError(f"Неизвестная модель: {model_name}. Доступны: {self.available_models}")

        self.model = MODEL_FACTORIES[model_name](classes=classes)

        if model_path:
            self.load_weights(model_path)

        self.epochs = epochs
        self.metric_names = metric_names

        # Аугментации нужны всегда (и для трейна, и для инференса)
        self.train_augmentation = get_imagenet_encoder_augmentation(phase='train')
        self.val_augmentation = get_imagenet_encoder_augmentation(phase='valid')

        # 2. Датасеты и Лоадеры
        self.train_loader = None
        self.val_loader = None
        self.train_dataset = None
        self.val_dataset = None

        # Проверяем наличие обязательных параметров для train
        if train_img_dirs and train_mask_dirs:
            if len(train_img_dirs) != len(train_mask_dirs):
                raise ValueError("Количество папок с изображениями и масками для обучения должно совпадать.")

            # Режим работы с множественными папками
            if val_img_dirs is None or val_mask_dirs is None:
                # Автоматическое разделение на train/val из всех указанных папок
                self.train_dataset, self.val_dataset = create_train_val_datasets_from_multiple_dirs(
                    img_dirs=train_img_dirs,
                    mask_dirs=train_mask_dirs,
                    train_ratio=train_ratio,
                    seed=train_val_seed,
                    train_augmentation=self.train_augmentation,
                    val_augmentation=self.val_augmentation,
                    mask_suffix=mask_suffix
                )
            else:
                if len(val_img_dirs) != len(val_mask_dirs):
                    raise ValueError("Количество папок с изображениями и масками для валидации должно совпадать.")

                # Раздельные папки для train и val
                self.train_dataset = MultiDirsDataset(
                    img_dirs=train_img_dirs,
                    mask_dirs=train_mask_dirs,
                    augmentation=self.train_augmentation,
                    mask_suffix=mask_suffix
                )
                self.val_dataset = MultiDirsDataset(
                    img_dirs=val_img_dirs,
                    mask_dirs=val_mask_dirs,
                    augmentation=self.val_augmentation,
                    mask_suffix=mask_suffix
                )

            # Создаем лоадеры если датасеты существуют
            if self.train_dataset:
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

        # 3. Оптимизатор, Лосс и Трейнер (только если есть данные и имя лосса)
        self.optimizer = None
        self.loss_fn = None
        self.scheduler = None
        self.trainer = None

        if self.train_loader and loss_name:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

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

        self.train_logs = None

    @property
    def available_models(self) -> list[str]:
        return list(MODEL_FACTORIES.keys())

    @property
    def available_losses(self) -> list[str]:
        return list(LOSS_FACTORIES.keys())

    def load_weights(self, path: str):
        """Загрузка весов модели"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл модели не найден: {path}")

        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        print(f'Веса загружены из {path}')

    def fit(
            self,
            save_model_path: str = './models/best_model.pth',
            save_logs_path: str = './logs/training_logs.json',
            save_checkpoint_path: str = None,
            load_checkpoint_path: str = None
    ):
        if self.trainer is None:
            raise RuntimeError(
                "Невозможно запустить fit(): не инициализирован Trainer. "
                "Проверьте наличие train_img_dirs, train_mask_dirs и loss_name.")

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

    def draw_logs(
            self,
            logs_path: str = None,
            show_metrics: bool = True,
            metric_names: list[str] = None
    ):
        if metric_names is None:
            metric_names = self.metric_names

        plot_epochs_data(
            self.train_logs,
            logs_path=logs_path,
            show_metrics=show_metrics,
            metric_names=metric_names
        )

    def predict_single(
            self,
            image_path: str,
            threshold: float = 0.5,
            visualize: bool = True,
            save_concat_path: str = None,
            save_mask_path: str = None
    ):
        """Предсказание для одного изображения."""
        self.model.eval()

        image_rgb, input_tensor, original_size = preprocess_image_for_model(
            image_path,
            self.val_augmentation,
            self.device
        )

        with torch.no_grad():
            output = self.model(input_tensor)

        binary_mask = postprocess_prediction(output, threshold, original_size=original_size)

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

        return binary_mask

    def predict_folder(
            self,
            input_dir: str,
            output_dir: str = './predictions',
            threshold: float = 0.5,
            save_masks: bool = True,
            save_concat: bool = True,
            visualize_console: bool = False,
            mask_suffix: str = '_mask',
            viz_suffix: str = '_viz'
    ):
        """Пакетное предсказание для всех изображений в папке."""
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
