from zenml import step
from pyspark.sql import DataFrame
import pandas as pd
from typing import Iterator
import logging
from src.domain.documents import TelegramChatDocument
from src.domain.schemas import pydantic_to_spark_schema
from datetime import datetime

# Logic extracted from your class to a standalone function for serialization
def _worker_validate_batch(iterator: Iterator[DataFrame]) -> Iterator[DataFrame]:
        """
        Worker: Replicates the original 'process_telegram_chat_data' logic 
        PRIOR to Pydantic validation.
        """
        #add logging
        
        logging.info("Starting worker_validate_batch")
        
        
        message_count = 0
        #print(f"numbers of batches: {len(list(iterator))}")
        
        for pdf in iterator:
            valid_rows = []
            for _, row in pdf.iterrows():
                try:
                    # 1. Capture Raw Context (chat_ctx)
                    msg         = row['raw_msg']
                    chat_id     = row['chat_id']
                    chat_name   = row['chat_name']
                    
                    # 2. Logic: ID & Sender extraction (matches original snippet)
                    raw_msg_id = msg.get('id')
                    if not raw_msg_id: continue
                    
                    message_uid = str(f"{chat_id}_{str(int(raw_msg_id))}")
                    sender_name = msg.get('from') if msg.get('from') else None
                    sender_id = msg.get('from_id').replace('user','') if msg.get('from_id') else None
                    text_msg = msg.get('text', "")
                    
                    # Transformation: replace('user','')
                    safe_reply_id = int(msg.get('reply_to_message_id')) if msg.get('reply_to_message_id') else None
                    
                    # Exact date logic from original code
                    raw_date = msg['date']
                    dt_obj = datetime.fromisoformat(raw_date.replace('Z', '+00:00'))
                    
                    date_unixtime = str(int(dt_obj.timestamp()))
                    week_id = f"{chat_id}_{dt_obj.strftime('%Y-%W')}"
                    
                    # 3. Logic: Filter condition (matches original)
                    if sender_name and sender_id and msg.get('type') != 'service':
                        message_count += 1
                        # 5. Instantiate Pydantic (Strict Validation)
                        # This ensures the 'flatten' validator handles the text field correctly
                        doc = TelegramChatDocument(
                            id=message_uid,
                            message_id=int(raw_msg_id),
                            chat_id=chat_id,
                            chat_name=chat_name,
                            date=raw_date,
                            date_unixtime=date_unixtime,
                            week_id=week_id,
                            sender=sender_name,
                            from_id=int(sender_id),
                            text=text_msg,
                            reactions=msg.get('reactions', []),
                            reply_to_message_id=safe_reply_id
                        )
                        
                        # Add to batch
                        valid_rows.append(doc.model_dump())
                       
              
                except Exception as e:
                    # Skip invalid messages silently like original code
                    # add error logging 
                    logging.error(f"Error processing message {raw_msg_id}: {e}")                    
                    continue
                    
            yield pd.DataFrame(valid_rows)

@step
def validate_messages(df_bronze: DataFrame) -> DataFrame:
    """
    Applies the validation worker to the bronze DataFrame.
    """
    schema=pydantic_to_spark_schema(TelegramChatDocument)
    return df_bronze.mapInPandas(
        _worker_validate_batch,
        schema=schema
    )