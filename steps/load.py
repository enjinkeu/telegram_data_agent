import logging
from pyspark.sql import DataFrame
from zenml import step, get_step_context

# Setup Logger
logger = logging.getLogger(__name__)

def _write_layer(df: DataFrame, db_name: str, collection: str) -> int:
    """
    Helper to write a specific dataframe to MongoDB with robust logging and error checking.
    """
    # 1. Validation
    if df is None:
        logger.warning(f"[{collection}] Skipped: DataFrame is None")
        return 0
        
    # 2. Observability: Count rows
    # We cache here because we are triggering an action (count) and then another action (write).
    # Without cache, Spark might re-compute the transformation DAG twice.
    df.cache()
    count = df.count()
    
    if count == 0:
        logger.info(f"[{collection}] Skipped: 0 records to write.")
        return 0
        
    # 3. Execution
    try:
        logger.info(f"[{collection}] Writing {count} records to {db_name}.{collection}...")
        df.write \
            .format("mongodb") \
            .mode("append") \
            .option("database", db_name) \
            .option("collection", collection) \
            .save()
        logger.info(f"[{collection}] Success.")
        return count
    except Exception as e:
        # Log context before crashing so you know WHICH collection failed
        logger.error(f"[{collection}] Failed to write to MongoDB: {e}")
        raise e

@step
def load_to_mongodb(df_users: DataFrame, df_threads: DataFrame, df_messages: DataFrame, db_name: str) -> None:
    """
    Loads processed DataFrames to MongoDB.
    Includes Row Counting, Empty Checks, Error Isolation, and ZenML Metadata tracking.
    """
    step_name = "load_to_mongodb"
    logger.info(f"[{step_name}] Starting load process to database: {db_name}")

    try:
        # 1. Write Collections
        users_count = _write_layer(df_users, db_name, "users")
        threads_count = _write_layer(df_threads, db_name, "threads")
        msgs_count = _write_layer(df_messages, db_name, "messages")
        
        # 2. Push Metadata to ZenML Dashboard
        # This allows you to track data volume trends over time in the UI
        try:
            step_context = get_step_context()
            step_context.add_output_metadata(
                output_name="output", # Even though this step returns None, we attach to the default output context
                metadata={
                    "loaded_users_count": users_count,
                    "loaded_threads_count": threads_count,
                    "loaded_messages_count": msgs_count,
                    "target_database": db_name
                }
            )
        except Exception as e:
            logger.warning(f"[{step_name}] Could not attach ZenML metadata: {e}")

        # 3. Final Summary Log
        logger.info(f"[{step_name}] Load Complete. Users: {users_count} | Threads: {threads_count} | Messages: {msgs_count}")
        
    except Exception as e:
        logger.critical(f"[{step_name}] Critical Failure during load: {e}")
        raise e