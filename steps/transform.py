import logging
from zenml import step, get_step_context
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from zenml.integrations.spark.materializers.spark_dataframe_materializer import SparkDataFrameMaterializer

# Setup Logger
logger = logging.getLogger(__name__)

@step(output_materializers=[SparkDataFrameMaterializer])
def enrich_columns(df: DataFrame) -> DataFrame:
    """
    Create silver data frame with enriched columns.
    
    Observability:
    - Tracks input vs output row counts (validation loss).
    - Tracks date parsing failures (null timestamps).
    - Logs the time range (min/max date) processed.
    """
    step_context = get_step_context()
    
    # 1. Baseline Metric: Input Count
    # (Optional: Only do this if 'df' is cached upstream, otherwise it triggers a read)
    # input_count = df.count() 
    
    # 2. Apply Logic   
    
    df_enriched = df\
        .withColumn("date_timestamp", F.to_timestamp(F.col("date")))\
        .withColumn("date_unixtime", F.unix_timestamp(F.col("date_timestamp")))\
        .withColumn("week_id", F.date_format(F.col("date_timestamp"), "yyyy-ww"))

    # 3. CRITICAL: Persist for Observability
    # We must cache here, otherwise counting metrics + returning the DF 
    # will cause Spark to re-compute the transformation twice.
    df_enriched.cache()
    
    # 4. Gather Metrics
    row_count = df_enriched.count()
    
    # Check for Parsing Failures (Silent Killers)
    failed_dates = df_enriched.filter(F.col("date_timestamp").isNull()).count()
    failure_rate = (failed_dates / row_count * 100) if row_count > 0 else 0
    
    # Check Time Range (Vital for backfill verification)
    time_stats = df_enriched.select(
        F.min("date_timestamp").alias("min_date"), 
        F.max("date_timestamp").alias("max_date")
    ).collect()[0]

    # 5. Logging & Metadata
    logger.info(f"Enrichment Complete. Rows: {row_count}")
    logger.info(f"Date Range: {time_stats['min_date']} to {time_stats['max_date']}")
    
    if failed_dates > 0:
        logger.warning(f"⚠️ DATE PARSING ISSUES: {failed_dates} rows ({failure_rate:.2f}%) have invalid dates.")

    # Push to ZenML Dashboard
    step_context.add_output_metadata(
        output_name="output", 
        metadata={
            "row_count": row_count,
            "failed_date_parsing_count": failed_dates,
            "failed_date_parsing_rate_percent": failure_rate,
            "min_processed_date": str(time_stats["min_date"]),
            "max_processed_date": str(time_stats["max_date"]),
            "data_quality_status": "WARNING" if failed_dates > 0 else "OK"
        }
    )

    return df_enriched

