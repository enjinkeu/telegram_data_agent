from pydantic_settings import BaseSettings

from typing import Set
from zenml.client import Client
from zenml.exceptions import EntityExistsError
from loguru import logger

class AppSettings(BaseSettings):
    # MongoDB Settings
    MONGO_USER: str = "llm_engineering"
    MONGO_PASS: str = "llm_engineering"
    MONGO_HOST: str = "127.0.0.1"
    MONGO_PORT: str = "27017"
    DB_NAME: str = "telegram_analytics"
    
    # uri="bolt://localhost:7687", user="neo4j", password="llm_engineering")
    NEON4J_URI: str = "bolt://localhost:7687"
    NEON4J_USER: str = "neo4j"
    NEON4J_PASS: str = "llm_engineering"
    
    # Spark Settings
    SPARK_APP_NAME: str = "TelegramIngestionPipeline"
    SPARK_MASTER: str = "local[*]"
    SPARK_MEMORY: str = "4g"
    
    # Input Data Settings
    INPUT_PATH: str = "../data/result.json"
    TARGET_CHATS: Set[str] = set()  # Empty set means process all chats

    @property
    def mongo_uri(self) -> str:
        return f"mongodb://{self.MONGO_USER}:{self.MONGO_PASS}@{self.MONGO_HOST}:{self.MONGO_PORT}/{self.DB_NAME}?authSource=admin&directConnection=true"

    # class Config:
    #     env_file = ".env"
    #     env_file_encoding = "utf-8"
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