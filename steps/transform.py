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


# --- Main Transform Steps ---



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
        
def _worker_thread_engine(pdf: DataFrame) -> DataFrame:
        """
        Builds threads and returns them with correct Types for Arrow serialization.
        """

        tracker = ThreadTracker()
        
        # 1. SORT & SANITIZE
        pdf_sorted = pdf.sort_values(by="message_id")
        
        # Fix IDs (Float -> Int)
        cols_to_fix = ['reply_to_message_id', 'message_id', 'from_id', 'chat_id']
        for col in cols_to_fix:
            if col in pdf_sorted.columns:
                pdf_sorted[col] = pdf_sorted[col].fillna(-1).astype(int)

        # Fix Dates (Int -> String)
        if 'date_unixtime' in pdf_sorted.columns:
            pdf_sorted['date_unixtime'] = pdf_sorted['date_unixtime'].astype(str)

        # 2. INGESTION LOOP
        for _, row in pdf_sorted.iterrows():
            try:
                row_dict = row.to_dict()
                
                # Revert Sentinels
                if row_dict.get('reply_to_message_id') == -1:
                    row_dict['reply_to_message_id'] = None
                
                # Clean Spark Cols
                for c in ['timestamp_dt', 'date_timestamp']:
                    if c in row_dict: del row_dict[c]
                
                msg = TelegramChatDocument(**row_dict)
                tracker.process_item(msg)
                
            except Exception:
                continue

        # 3. EXTRACTION LOOP
        results = []
        for t_id, thread_doc in tracker.threads.items():
            if len(thread_doc.messages) < 2: continue

            # Build Nested Messages
            nested_msgs = []
            for m in thread_doc.messages:
                # Transform Reactions to match REACTION_SCHEMA
                # Pydantic model: {'emoji': '👍', 'count': 1}
                # Schema: StructType(emoji=String, count=Long) -> Perfect Match
                clean_reactions = [r.model_dump() for r in m.reactions] if m.reactions else []

                nested_msgs.append({
                    "id": str(m.id),
                    "message_id": int(m.message_id),
                    "date": str(m.date),
                    "date_unixtime": str(m.date_unixtime),
                    "sender": str(m.sender),
                    "from_id": int(m.from_id),
                    "text": str(m.text),
                    "reactions": clean_reactions
                })

            results.append({
                "thread_id": str(t_id),
                "chat_id": int(thread_doc.chat_id),
                "chat_name": str(thread_doc.chat_name),
                "week_id": str(thread_doc.week_id),
                "root_message_id": int(thread_doc.root_message_id),
                "transcript": str(thread_doc.render_for_llm()),
                "messages": nested_msgs,
                "message_count": len(thread_doc.messages),
                "participant_count": len(thread_doc.participant_ids),
                "start_unixtime": int(thread_doc.start_unixtime) if thread_doc.start_unixtime != float('inf') else 0,
                "end_unixtime": int(thread_doc.end_unixtime) if thread_doc.end_unixtime != float('-inf') else 0
            })
            
        return pd.DataFrame(results)

@step
def aggregate_users(df: DataFrame) -> DataFrame:
    return enrich_columns(df).select(
        F.col("from_id").alias("telegram_id"),
        F.col("sender").alias("display_name")
    ).distinct().filter(F.col("telegram_id").isNotNull())

@step
def reconstruct_threads(df: DataFrame) -> DataFrame:
   
    df_silver = enrich_columns(df)
    return df_silver.groupBy("chat_id").applyInPandas(
        _worker_thread_engine,
        schema=GOLD_THREAD_SCHEMA
    )