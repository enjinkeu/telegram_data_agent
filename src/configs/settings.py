from pydantic import BaseSettings
from typing import Set
from zenml.client import Client
from zenml.exceptions import EntityExistsError
from loguru import logger

class AppSettings(BaseSettings):
    # MongoDB Settings
    MONGO_USER: str = "llm_engineering"
    MONGO_PASS: str = "llm_engineering"
    MONGO_HOST: str = "mongo"
    MONGO_PORT: int = 27017
    DB_NAME: str = "telegram_analytics"
    
    # Spark Settings
    SPARK_APP_NAME: str = "TelegramIngestionPipeline"
    SPARK_MASTER: str = "local[*]"
    SPARK_MEMORY: str = "4g"
    
    # Input Data Settings
    INPUT_PATH: str = "../data/telegram_export.json"
    TARGET_CHATS: Set[str] = set()  # Empty set means process all chats

    @property
    def mongo_uri(self) -> str:
        return f"mongodb://{self.MONGO_USER}:{self.MONGO_PASS}@{self.MONGO_HOST}:{self.MONGO_PORT}/{self.DB_NAME}?authSource=admin"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Map nested yaml structure if needed, or prefer flattening env vars
        
    def export(self) -> None:
        """
        Exports the settings to the ZenML secret store.
        """

        env_vars = settings.model_dump()
        for key, value in env_vars.items():
            env_vars[key] = str(value)

        client = Client()

        try:
            client.create_secret(name="settings", values=env_vars)
        except EntityExistsError:
            logger.warning(
                "Secret 'scope' already exists. Delete it manually by running 'zenml secret delete settings', before trying to recreate it."
            )

        
#settings = Settings.load_settings()
settings = AppSettings()