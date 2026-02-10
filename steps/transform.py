import pandas as pd
from typing import Iterator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, ArrayType
import logging
from datetime import datetime

from zenml import step
from src.domain.documents import TelegramChatDocument, ThreadTracker
from src.domain.schemas import GOLD_THREAD_SCHEMA
from .validate import validate_messages
# --- Worker Functions (Must be static/top-level for pickling) ---
from zenml.integrations.spark.materializers.spark_dataframe_materializer import SparkDataFrameMaterializer

# --- Main Transform Steps ---


@step(output_materializers=[SparkDataFrameMaterializer])
def enrich_columns(df: DataFrame) -> DataFrame:
    """
    Create silver data frame with enriched columns: date_timestamp, date_unixtime, week_id.
    
    :param df: Bronze DataFrame with raw Telegram messages
    
    """
    df_validated = validate_messages(df)
    return df_validated\
        .withColumn("date_timestamp", F.to_timestamp(F.col("date")))\
        .withColumn("date_unixtime", F.unix_timestamp(F.col("date_timestamp")))\
        .withColumn("week_id", F.date_format(F.col("date_timestamp"), "yyyy-ww"))
        

