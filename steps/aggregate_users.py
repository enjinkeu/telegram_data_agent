import logging
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from zenml import step, get_step_context
from zenml.integrations.spark.materializers.spark_dataframe_materializer import SparkDataFrameMaterializer

# Configure logger
logger = logging.getLogger(__name__)

@step(output_materializers=[SparkDataFrameMaterializer])
def aggregate_users(df: DataFrame) -> DataFrame:
    """
    Extracts a unique list of users from the message DataFrame.
    """
    step_name = "aggregate_users"
    
    # 1. Filter
    # NOTE: We removed (F.col("type") != "service") because the upstream 'validate' 
    # step already filtered these rows and dropped the 'type' column.
    df_filtered = df.filter(
        (F.col("from_id").isNotNull()) & 
        (F.col("sender").isNotNull()) & 
        (F.col("sender") != "")
    )

    # 2. Select and Cast Columns
    # Legacy logic: telegram_id = sender_id (as string), display_name = sender_name
    df_users = df_filtered.select(
        F.col("from_id").cast("string").alias("telegram_id"),
        F.col("sender").alias("display_name")
    ).distinct()

    # 3. Observability: Count & Log
    # This forces execution, ensuring we catch any data quality issues immediately.
    user_count = df_users.count()

    try:
        step_context = get_step_context()
        step_context.add_output_metadata(
            output_name="output", 
            metadata={"distinct_user_count": user_count}
        )
    except Exception as e:
        logger.warning(f"[{step_name}] Could not attach ZenML metadata: {e}")

    logger.info(f"[{step_name}] Successfully aggregated {user_count} distinct users.")
    
    if user_count == 0:
        logger.warning(f"[{step_name}] WARNING: 0 users found. Check upstream 'from_id' population.")

    return df_users