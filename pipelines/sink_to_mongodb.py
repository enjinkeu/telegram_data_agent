
from steps import ingest, validate, aggregate_users, reconstruct_threads,load, transform
from zenml import pipeline  
from src.configs import settings


@pipeline
def telegram_data_etl(input_path: str , target_chats: list) -> None:
    # Ingest Step
    df_raw = ingest.ingest_json_data(input_path=input_path, target_chats=target_chats)
    
    # Validate Step
    df_messages = validate.validate_messages(df_raw)
    
    # 2. Add columns ONCE (move enrich_columns logic to a standalone @step)
    df_silver = transform.enrich_columns(df_messages)
    
    
    # 3. Use df_silver for both downstream steps
    df_users = aggregate_users.aggregate_users(df_silver)
    df_threads = reconstruct_threads.reconstruct_threads(df_silver)
   
       
    # Load Step
    load.load_to_mongodb(df_users, df_threads, df_messages, db_name="telegram_analytics")



