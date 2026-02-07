
import steps.ingest as ingest
import steps.transform as transform
import steps.validate as validate
import steps.load as load_to_mongodb
from zenml import pipeline  
from src.configs import settings

@pipeline
def telegram_data_etl() -> None:
    # Ingest Step
    df_raw = ingest.ingest_json_data(input_path=settings.INPUT_DATA_PATH, target_chats=list(settings.TARGET_CHATS))
    
    # Validate Step
    df_validated = validate.validate_messages(df_raw)
    
    # Transform Step
    df_users, df_threads, df_messages = transform.transform_data(df_validated)
    
    # Load Step
    load_to_mongodb.load_to_mongodb(df_users, df_threads, df_messages)



