from pyspark.sql import SparkSession
from src.configs.settings import settings

def get_spark_session(app_name: str = "TelegramETL") -> SparkSession:
    return SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.3.0") \
        .config("spark.mongodb.read.connection.uri", settings.mongo_uri) \
        .config("spark.mongodb.write.connection.uri", settings.mongo_uri) \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
        .getOrCreate()