from typing import Type, get_origin, get_args, Union, List, Dict, Any
from pyspark.sql.types import (
    StructType, StructField, StringType, LongType, 
    DoubleType, BooleanType, ArrayType, MapType
)
from pydantic import BaseModel
import collections

 # Define Schemas (Reaction, Nested, Gold Thread) here as per your original code
    # 1. Define Reaction Schema explicitly (Fixes the Int/String crash)
REACTION_SCHEMA = StructType([
            StructField("emoji", StringType(), True),
            StructField("count", LongType(), True)  # Now properly supports Integers
        ])
BRONZE_SCHEMA = StructType([
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
    # 2. Nested Message Schema uses the Reaction Schema
NESTED_MESSAGE_SCHEMA = StructType([
            StructField("id", StringType(), True),
            StructField("message_id", LongType(), True),
            StructField("date", StringType(), True),
            StructField("date_unixtime", StringType(), True), 
            StructField("sender", StringType(), True),
            StructField("from_id", LongType(), True),
            StructField("text", StringType(), True),
            # UPDATED: Use Array of Structs instead of Map
            StructField("reactions", ArrayType(REACTION_SCHEMA), True) 
        ])
        
    # The Output Schema for Step 5 (Thread Reconstruction)
GOLD_THREAD_SCHEMA = StructType([
            StructField("thread_id", StringType(), True),
            StructField("chat_id", LongType(), True),
            StructField("chat_name", StringType(), True),
            StructField("week_id", StringType(), True),
            StructField("root_message_id", LongType(), True),
            
            # 1. The Transcript (For LLM / RAG)
            StructField("transcript", StringType(), True),
            
            # 2. The Structured Data (For Database / Analytics)
            StructField("messages", ArrayType(NESTED_MESSAGE_SCHEMA), True),
            
            # 3. Metadata
            StructField("message_count", LongType(), True),
            StructField("participant_count", LongType(), True),
            StructField("start_unixtime", LongType(), True),
            StructField("end_unixtime", LongType(), True)
        ])

def _get_simple_spark_type(py_type: Type) -> Any:
    """Helper to map Python primitives to Spark types."""
    if py_type == int: return LongType() 
    elif py_type == str: return StringType()
    elif py_type == float: return DoubleType()
    elif py_type == bool: return BooleanType()
    return StringType()

def pydantic_to_spark_schema(pydantic_model: Type[BaseModel]) -> StructType:
    """Recursively converts a Pydantic model to a PySpark StructType schema (Pydantic V1 compatible)."""
    fields = []
    
    # FIX: Use __fields__ instead of model_fields for Pydantic V1
    for name, field in pydantic_model.__fields__.items():
        # FIX: In V1, the type is stored in 'outer_type_'
        py_type = field.outer_type_
        
        # Check if field is optional
        nullable = not field.required
        
        origin = get_origin(py_type)
        args = get_args(py_type)
        
        # Handle Optional[T] which is Union[T, NoneType]
        if origin == Union and type(None) in args:
            nullable = True
            # Get the actual type (e.g., extract int from Optional[int])
            non_none_args = [arg for arg in args if arg is not type(None)]
            if non_none_args:
                py_type = non_none_args[0]
                origin = get_origin(py_type)
                args = get_args(py_type)

        spark_type = StringType() # Default fallback

        # 1. Handle Lists
        if origin is list or origin is List:
            inner_type = args[0] if args else str
            # Check if the inner type is a Pydantic Model
            if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
                spark_type = ArrayType(pydantic_to_spark_schema(inner_type), True)
            else:
                spark_type = ArrayType(_get_simple_spark_type(inner_type), True)
        
        # 2. Handle Nested Pydantic Models
        elif isinstance(py_type, type) and issubclass(py_type, BaseModel):
            spark_type = pydantic_to_spark_schema(py_type)
            
        # 3. Handle Simple Types
        else:
            spark_type = _get_simple_spark_type(py_type)
            
        fields.append(StructField(name, spark_type, nullable))
        
    return StructType(fields)