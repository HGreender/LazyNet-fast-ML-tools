import os
import mlflow


class MLflowManager:
    def __init__(self):
        self.MLFLOW_TRACKING_URI = 'http://192.168.0.65:5000'
        self.MLFLOW_S3_ENDPOINT_URL = 'http://192.168.0.65:5001'  # Адрес MinIO API
        self.AWS_ACCESS_KEY_ID = 'minioadmin'
        self.AWS_SECRET_ACCESS_KEY = 'Gjkeghbynth2025m'
        self.AWS_DEFAULT_REGION = 'us-east-1'

    def quick_init(self, experiment_name: str):
        """Настройка окружения и подключение к серверу"""
        os.environ['MLFLOW_TRACKING_URI'] = self.MLFLOW_TRACKING_URI
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = self.MLFLOW_S3_ENDPOINT_URL
        os.environ['AWS_ACCESS_KEY_ID'] = self.AWS_ACCESS_KEY_ID
        os.environ['AWS_SECRET_ACCESS_KEY'] = self.AWS_SECRET_ACCESS_KEY
        os.environ['AWS_DEFAULT_REGION'] = self.AWS_DEFAULT_REGION

        mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)

        # Проверка существования эксперимента или создание нового
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)

        mlflow.set_experiment(experiment_name)
        print(f"✅ MLflow initialized. Experiment: {experiment_name}")
        print(f"📦 Artifacts will be stored in MinIO at {self.MLFLOW_S3_ENDPOINT_URL}")

    def start_run(self, run_name=None, tags=None):
        """Контекстный менеджер для запуска эксперимента"""
        return mlflow.start_run(run_name=run_name, tags=tags)