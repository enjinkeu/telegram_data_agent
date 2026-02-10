import logging
from zenml import step, get_step_context
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, ArrayType, MapType
from src.infrastructure.spark import get_spark_session
from zenml.integrations.spark.materializers.spark_dataframe_materializer import SparkDataFrameMaterializer

# Configure standard Python logging to show up in ZenML logs
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@step(output_materializers=[SparkDataFrameMaterializer])
def ingest_json_data(input_path: str, target_chats: list) -> DataFrame:
    step_context = get_step_context()
    spark = get_spark_session()
    
    # 1. Start Observability: Log configs
    logger.info(f"Starting ingestion for target chats: {target_chats}")
    #step_context.add_output_metadata(output_name="output", metadata={"target_chats": target_chats})

    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    spark.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")
    
    # ... [Schema Definition remains the same] ...
    bronze_schema = StructType([
        StructField("chats", StructType([
            StructField("list", ArrayType(StructType([
                StructField("id", LongType(), True),
                StructField("name", StringType(), True),
                StructField("messages", ArrayType(StructType([
                    StructField("id", LongType(), True),
                    StructField("type", StringType(), True),
                    StructField("date", StringType(), True),
                    StructField("from", StringType(), True),
                    StructField("from_id", StringType(), True), 
                    StructField("text", StringType(), True),
                    StructField("reply_to_message_id", LongType(), True),
                    StructField("reactions", ArrayType(MapType(StringType(), StringType())), True)
                ])), True)
            ])), True)
        ]), True)
    ])

    # 2. Read Data
    df_raw = spark.read.schema(bronze_schema).option("multiline", "true").json(input_path)
    
    # 3. Filter and Explode
    df_filtered = df_raw.select(F.explode("chats.list").alias("chat")) \
        .filter(F.col("chat.name").isin(list(target_chats)))
        
    df_exploded = df_filtered.select(
        F.col("chat.id").alias("chat_id"),
        F.col("chat.name").alias("chat_name"),
        F.explode("chat.messages").alias("raw_msg")
    )

    # 4. CRITICAL: Cache the result to prevent re-computation during counting
    # This keeps the 'observability tax' low.
    df_cached = df_exploded.repartition(8).cache()

    # 5. Observability: Materialize metrics
    row_count = df_cached.count()
    
    # Check for silent failures (Schema mismatches resulting in Null IDs)
    null_id_count = df_cached.filter(F.col("chat_id").isNull()).count()
    
    # Get distinct chats actually found vs requested
    found_chats = [r['chat_name'] for r in df_cached.select("chat_name").distinct().collect()]
    missing_chats = set(target_chats) - set(found_chats)

    # 6. Log Metrics to ZenML & Console
    logger.info(f"Ingestion Complete. Rows: {row_count}. Null IDs: {null_id_count}")
    
    if missing_chats:
        logger.warning(f"⚠️ MISSING CHATS: The following target chats were not found in data: {missing_chats}")

    # Push metadata to ZenML Dashboard
    step_context.add_output_metadata(
        output_name="output", 
        metadata={
            "row_count": row_count,
            "null_ids": null_id_count,
            "found_chats": found_chats,
            "missing_chats": list(missing_chats),
            "spark_partitions": df_cached.rdd.getNumPartitions()
        }
    )

    return df_cached