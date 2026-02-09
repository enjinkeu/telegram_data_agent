
from steps import ingest, validate, aggregate_users, reconstruct_threads,load
from zenml import pipeline  
from src.configs import settings


@pipeline
def telegram_data_etl(input_path: str , target_chats: list) -> None:
    # Ingest Step
    df_raw = ingest.ingest_json_data(input_path=input_path, target_chats=target_chats)
    
    # Validate Step
    df_messages = validate.validate_messages(df_raw)
    
    #get users and threads
    df_users = aggregate_users.aggregate_users(df_messages)
    df_threads = reconstruct_threads.reconstruct_threads(df_messages)
   
       
    # Load Step
    load.load_to_mongodb(df_users, df_threads, df_messages)



