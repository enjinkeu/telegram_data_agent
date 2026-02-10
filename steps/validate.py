import json
from typing import NamedTuple, Iterator
from pyspark.sql import DataFrame, Row
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, IntegerType
)
from zenml import step, get_step_context
from zenml.integrations.spark.materializers.spark_dataframe_materializer import SparkDataFrameMaterializer



@step(output_materializers=[SparkDataFrameMaterializer])
def flatten_telegram_data(bronze_df: DataFrame) -> DataFrame:
    """
    Flattens the Bronze DataFrame into a clean Silver table schema.
    Skipping Pydantic validation to avoid serialization overhead/errors.
    """
    step_context = get_step_context()
    
    # 1. Native Spark Flattening
    # This executes entirely within the JVM, avoiding Python pickling issues
    df_silver = bronze_df.select(
        # Identity
        F.concat_ws("_", F.col("chat_id"), F.col("raw_msg.id")).alias("id"),
        F.col("raw_msg.id").cast(LongType()).alias("message_id"),
        F.col("chat_id").cast(LongType()).alias("chat_id"),
        F.col("chat_name"),
        
        # Time
        F.col("raw_msg.date").alias("date"),
        
        # Content
        # We handle the reserved keyword 'from' by aliasing immediately
        F.col("raw_msg.from").alias("sender"),
        F.col("raw_msg.from_id").alias("from_id"),
        
        # Data Fields
        # We cast text to String to ensure consistent types if the input is mixed
        F.col("raw_msg.text").cast(StringType()).alias("text"),
        
        # Reactions are usually complex structures; we keep them as-is or cast to JSON string
        F.to_json(F.col("raw_msg.reactions")).alias("reactions"),
        
        F.col("raw_msg.reply_to_message_id").alias("reply_to_message_id")
    )

    # 2. Basic Observability
    # Persist the dataframe to avoid re-reading for the count
    df_silver.cache()
    row_count = df_silver.count()
    
    step_context.add_output_metadata(
        output_name="output", 
        metadata={
            "row_count": row_count,
            "status": "flattened_no_validation"
        }
    )

    return df_silver