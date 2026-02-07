from zenml import step
from pyspark.sql import DataFrame

@step
def load_to_mongodb(df_users: DataFrame, df_threads: DataFrame, df_messages: DataFrame) -> None:
    # Write Users
    df_users.write \
        .format("mongodb") \
        .mode("append") \
        .option("database", "telegram_analytics") \
        .option("collection", "users") \
        .save()

    # Write Threads
    df_threads.write \
        .format("mongodb") \
        .mode("append") \
        .option("database", "telegram_analytics") \
        .option("collection", "threads") \
        .save()
        
    # Write Messages
    df_messages.write \
        .format("mongodb") \
        .mode("append") \
        .option("database", "telegram_analytics") \
        .option("collection", "messages") \
        .save()