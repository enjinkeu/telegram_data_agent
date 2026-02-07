from pydantic_settings import BaseSettings
from typing import Set

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
    INPUT_PATH: str
    TARGET_CHATS: Set[str]

    @property
    def mongo_uri(self) -> str:
        return f"mongodb://{self.MONGO_USER}:{self.MONGO_PASS}@{self.MONGO_HOST}:{self.MONGO_PORT}/{self.DB_NAME}?authSource=admin"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Map nested yaml structure if needed, or prefer flattening env vars
        
settings = AppSettings()