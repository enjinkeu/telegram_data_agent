from zenml import step
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, ArrayType, MapType
from src.infrastructure.spark import get_spark_session
from src.configs import settings

@step
def ingest_json_data() -> DataFrame:
    spark = get_spark_session()
    
    # Define Bronze Schema (As per your code)
    bronze_schema = StructType([
        StructType([
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
    ])

    df_raw = spark.read.schema(bronze_schema).option("multiline", "true").json(settings.INPUT_DATA_PATH)
    
    # Filter and Explode
    df_filtered = df_raw.select(F.explode("chats.list").alias("chat")) \
        .filter(F.col("chat.name").isin(list(settings.TARGET_CHATS)))

    return df_filtered.select(
        F.col("chat.id").alias("chat_id"),
        F.col("chat.name").alias("chat_name"),
        F.explode("chat.messages").alias("raw_msg")
    )