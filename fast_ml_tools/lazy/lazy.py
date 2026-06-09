import os

from tqdm import tqdm
import cv2
import torch
from torch.utils.data import DataLoader

from fast_ml_tools.ml.models import efficientnetb4_unet, efficientnetb4_unetpp
from fast_ml_tools.ml.trainer import Trainer
from fast_ml_tools.ml.augmentations import get_imagenet_encoder_augmentation
from fast_ml_tools.ml.datasets import DirsDataset, create_train_val_datasets
from fast_ml_tools.ml.losses import DiceBCELoss, FocalLoss, DiceFocalLoss
from fast_ml_tools.visualization import plot_epochs_data, show_segmentation
from fast_ml_tools.ml.metrics import get_segmentation_metrics
from fast_ml_tools.ml.utils import preprocess_image_for_model, postprocess_prediction

MODEL_FACTORIES = {
    'efficientnetb4_unet': efficientnetb4_unet,
    'efficientnetb4_unet++': efficientnetb4_unetpp,
}

LOSS_FACTORIES = {
    'focal': FocalLoss,
    'dice_bce': DiceBCELoss,
    'dice_focal': DiceFocalLoss,
}


def _get_device(num: int = 0):
    """Получение устройства (CUDA/CPU)"""
    print(f"Доступно карт: {torch.cuda.device_count()}")
    if num >= torch.cuda.device_count() or num < 0:
        num = 0
        print("Выбрана дефолтная видеокарта")

    device = torch.device(f"cuda:{num}" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")
    return device


class LazyNet:
    ''' Класс для быстрого обучения моделей бинарной сегментации.
    '''

    def __init__(
            self,
            model_name: str,
            loss_name: str,
            train_img_dir: str,
            train_mask_dir: str,
            model_path: str = None,
            val_img_dir: str = None,
            val_mask_dir: str = None,
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
        self.device = _get_device(device_id)
        self.verbose = verbose

        self.model_path = model_path

        if model_name is None and model_path is not None:
            raise ValueError("Для загрузки модели (model_path) необходимо указать model_name, чтобы знать архитектуру.")
        if model_name not in MODEL_FACTORIES:
            raise ValueError(f"Неизвестная модель: {model_name}. Доступны: {list(MODEL_FACTORIES.keys())}")

        self.model = MODEL_FACTORIES[model_name](classes=classes)

        if model_path:
            self.load_weights(model_path)

        self.epochs = epochs

        self.train_augmentation = get_imagenet_encoder_augmentation(phase='train')
        self.val_augmentation = get_imagenet_encoder_augmentation(phase='valid')

        if val_img_dir is None or val_mask_dir is None:
            self.train_dataset, self.val_dataset = create_train_val_datasets(
                img_dir=train_img_dir,
                mask_dir=train_mask_dir,
                train_ratio=train_ratio,
                seed=train_val_seed,
                train_augmentation=self.train_augmentation,
                val_augmentation=self.val_augmentation
            )
        else:
            self.train_dataset = DirsDataset(train_img_dir, train_mask_dir, augmentation=self.train_augmentation)
            self.val_dataset = DirsDataset(val_img_dir, val_mask_dir, augmentation=self.val_augmentation)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # TODO: Добавить конфиги для функций ошибок?
        if loss_name not in LOSS_FACTORIES:
            raise ValueError(f"Неизвестная функция ошибки: {loss_name}. Доступны: {list(LOSS_FACTORIES.keys())}")
        self.loss_fn = LOSS_FACTORIES[loss_name]()

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=lr_factor,
            patience=lr_patience
        )
        self.early_stopping_threshold = early_stopping_threshold

        self.metric_names = metric_names
        metrics = get_segmentation_metrics(self.metric_names)

        self.trainer = Trainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            early_stopping_threshold=self.early_stopping_threshold,
            metrics=metrics,
            scheduler=self.scheduler,
            device=self.device,
            verbose=self.verbose,
        )

        self.train_logs = None

    def load_weights(self, path: str):
        """Загрузка весов модели"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл модели не найден: {path}")

        # map_location нужен, чтобы грузить на CPU если CUDA недоступна, или на конкретную GPU
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f'Веса загружены из {path}')

    def fit(
            self,
            save_model_path: str = './models/best_model.pth',
            save_logs_path: str = './logs/training_logs.json',
            save_checkpoint_path: str = None,
            load_checkpoint_path: str = None
    ):
        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
        os.makedirs(os.path.dirname(save_logs_path), exist_ok=True)

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
            visualize: bool = False,
            save_concat_path: str = None,
            save_mask_path: str = None
    ):
        """
        Предсказание для одного изображения.

        Args:
            image_path: Путь к исходному изображению.
            threshold: Порог бинаризации.
            visualize: Показать результат в окне (plt.show).
            save_concat_path: Сохранить картинку с наложением маски.
            save_mask_path: Сохранить чистую бинарную маску (PNG).
        """
        self.model.eval()

        image_rgb, input_tensor, original_size = preprocess_image_for_model(
            image_path,
            self.val_augmentation,
            self.device
        )

        with torch.no_grad():
            output = self.model(input_tensor)

        binary_mask = postprocess_prediction(output, threshold, original_size=original_size)

        # 1. Сохранение чистой маски (если нужно)
        if save_mask_path:
            os.makedirs(os.path.dirname(save_mask_path) or '.', exist_ok=True)
            # Сохраняем как черно-белое изображение
            cv2.imwrite(save_mask_path, binary_mask * 255)
            print(f'Mask saved to {save_mask_path}')

        # 2. Визуализация (если нужно показать или сохранить наложение)
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
            save_masks: bool = True,  # Сохранять ли чистые маски
            save_concat: bool = True,  # Сохранять ли визуализации
            visualize_console: bool = False,  # Показывать ли каждое окно в процессе (медленно!)
            mask_suffix: str = '_mask',
            viz_suffix: str = '_viz'
    ):
        """
        Пакетное предсказание для всех изображений в папке.
        Создает подпапки 'masks' и 'viz' внутри output_dir.
        """
        self.model.eval()

        # Создаем структуры папок
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
