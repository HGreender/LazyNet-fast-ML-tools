import optuna
import torch
import gc

from fast_ml_tools import LazyNet
from fast_ml_tools.lazy.lazy import MODEL_FACTORIES, LOSS_FACTORIES


class LazyNetOptuna:
    def __init__(
            self,
            # --- Данные ---
            train_img_dirs: list[str],
            train_mask_dirs: list[str],
            mask_suffix: str | list[str] = "_mask",

            # --- Общие настройки ---
            classes: list = ['target_class'],
            device_id: int = 0,
            num_workers: int = 0,
            epochs: int = 30,
            n_trials: int = 30,
            save_dir: str = './optuna_trials',
            train_ratio: float = 0.8,
            train_val_seed: int = 42,

            # --- Пространства поиска гиперпараметров (Search Spaces) ---
            models_space: list = MODEL_FACTORIES,
            losses_space: list = LOSS_FACTORIES,
            lr_range: tuple = (5e-6, 5e-4),  # (min, max)
            wd_range: tuple = (1e-6, 1e-4),  # (min, max)
            batch_sizes: list = [8, 16, 32],
            lr_factor_range: tuple = (0.1, 0.5),  # (min, max)
            lr_patience_range: tuple = (5, 15),  # (min, max)
    ):
        self.train_img_dirs = train_img_dirs
        self.train_mask_dirs = train_mask_dirs
        self.mask_suffix = mask_suffix
        self.classes = classes
        self.device_id = device_id
        self.num_workers = num_workers
        self.default_epochs = epochs
        self.n_trials = n_trials
        self.save_dir = save_dir
        self.train_ratio = train_ratio
        self.train_val_seed = train_val_seed

        # Сохраняем пространства поиска
        self.models_space = models_space
        self.losses_space = losses_space
        self.lr_range = lr_range
        self.wd_range = wd_range
        self.batch_sizes = batch_sizes
        self.lr_factor_range = lr_factor_range
        self.lr_patience_range = lr_patience_range

        self.study = None

    def _objective(self, trial):
        """Внутренняя функция-объектив"""
        lazy_net = None
        try:
            # 1. Выбор из заданных списков
            model_name = trial.suggest_categorical('model', self.models_space)
            loss_name = trial.suggest_categorical('loss', self.losses_space)
            batch_size = trial.suggest_categorical('batch_size', self.batch_sizes)

            # 2. Выбор из диапазонов (float/int)
            lr = trial.suggest_float('lr', self.lr_range[0], self.lr_range[1], log=True)
            weight_decay = trial.suggest_float('weight_decay', self.wd_range[0], self.wd_range[1], log=True)
            lr_factor = trial.suggest_float('lr_factor', self.lr_factor_range[0], self.lr_factor_range[1], step=0.1)
            lr_patience = trial.suggest_int('lr_patience', self.lr_patience_range[0], self.lr_patience_range[1])

            lazy_net = LazyNet(
                model_name=model_name,
                loss_name=loss_name,
                train_img_dirs=self.train_img_dirs,
                train_mask_dirs=self.train_mask_dirs,
                mask_suffix=self.mask_suffix,
                train_ratio=self.train_ratio,
                train_val_seed=self.train_val_seed,
                epochs=self.default_epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                lr_patience=lr_patience,
                lr_factor=lr_factor,
                verbose=False,
                device_id=self.device_id,
                num_workers=self.num_workers,
                classes=self.classes,
                metric_names=None
            )

            best_val_loss = lazy_net.fit_with_optuna(
                trial=trial,
                save_dir=self.save_dir,
                epochs=self.default_epochs
            )

            return best_val_loss

        finally:
            if lazy_net is not None:
                del lazy_net

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def run(self, direction='minimize', n_warmup_steps=5):
        """Запуск оптимизации"""
        self.study = optuna.create_study(
            direction=direction,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps)
        )

        self.study.optimize(
            self._objective,
            n_trials=self.n_trials,
            catch=(ValueError, RuntimeError, torch.cuda.OutOfMemoryError)
        )

        print(f"✅ Оптимизация завершена! Лучший результат: {self.study.best_value:.4f}")
        return self.study

    def get_best_params(self):
        if not self.study:
            raise RuntimeError("Сначала запустите метод run()")
        return self.study.best_params

    def plot_results(self):
        if not self.study:
            raise RuntimeError("Нет данных для визуализации")
        optuna.visualization.plot_optimization_history(self.study).show()
        optuna.visualization.plot_param_importances(self.study).show()
        optuna.visualization.plot_slice(self.study).show()
