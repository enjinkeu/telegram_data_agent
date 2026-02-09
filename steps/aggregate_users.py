import pandas as pd
from typing import Iterator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, ArrayType
import logging
from datetime import datetime

from zenml import step
from .transform import enrich_columns


@step
def aggregate_users(df: DataFrame) -> DataFrame:
    return enrich_columns(df).select(
        F.col("from_id").alias("telegram_id"),
        F.col("sender").alias("display_name")
    ).distinct().filter(F.col("telegram_id").isNotNull())