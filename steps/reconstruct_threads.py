import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType
from zenml import step, get_step_context
from zenml.integrations.spark.materializers.spark_dataframe_materializer import SparkDataFrameMaterializer
from .transform import enrich_columns

# Configure logger
logger = logging.getLogger(__name__)

# --- 1. CONFIGURATION ---

GOLD_THREAD_SCHEMA = StructType([
    StructField('thread_id', StringType(), True),
    StructField('chat_id', LongType(), True),
    StructField('chat_name', StringType(), True),
    StructField('week_id', StringType(), True),
    StructField('root_message_id', LongType(), True),
    StructField('transcript', StringType(), True),
    StructField('messages_json', StringType(), True),  # Flattened JSON string
    StructField('message_count', IntegerType(), True),
    StructField('participant_count', IntegerType(), True),
    StructField('start_unixtime', LongType(), True),
    StructField('end_unixtime', LongType(), True)
])

def _generate_transcript(chat_name: str, messages: list) -> str:
    """
    Generates a text transcript from a list of message dicts.
    Replaces the old Pydantic 'render_for_llm' method.
    """
    # 1. Sort chronologically
    sorted_msgs = sorted(messages, key=lambda x: int(x.get('date_unixtime', 0)))
    
    # 2. Metadata Header
    start = int(sorted_msgs[0]['date_unixtime'])
    end = int(sorted_msgs[-1]['date_unixtime'])
    duration_min = int((end - start) / 60)
    
    lines = [
        f"### CONVERSATION THREAD: {chat_name}",
        f"### METADATA: {len(messages)} msgs | {duration_min} mins duration",
        "### TRANSCRIPT: "
    ]

    # 3. Message Loop
    for msg in sorted_msgs:
        # Convert timestamp to readable string
        ts = int(msg.get('date_unixtime', 0))
        ts_str = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M')
        
        sender = msg.get('sender', 'Unknown')
        text = msg.get('text', '')
        
        # Format Reactions (if any exist)
        # Input format is expected to be a list of dicts: [{'emoji': '👍', 'count': 1}, ...]
        reacts = ""
        reactions = msg.get('reactions', [])
        if reactions:
            # Sort by count descending and take top 2
            try:
                top = sorted(reactions, key=lambda x: x.get('count', 0), reverse=True)[:2]
                react_str = ' '.join([f"{r.get('emoji')}{r.get('count')}" for r in top])
                reacts = f" {{rxn: {react_str}}}"
            except Exception:
                pass # Skip malformed reactions in transcript

        lines.append(f"[{ts_str}] {sender}: {text}{reacts}")

    return "\n".join(lines)

def _worker_thread_engine(pdf: DataFrame) -> DataFrame:
    """
    Worker: Reconstructs threads using pure Python dicts (No Pydantic).
    """
    if pdf.empty:
        return pd.DataFrame()
    
    chat_context = pdf['chat_id'].iloc[0] if 'chat_id' in pdf.columns else "unknown"
    logger.info(f"[Worker] Processing batch for chat_id={chat_context} | Rows: {len(pdf)}")

    # --- DATA STRUCTURES (The Logic Replacement) ---
    # Threads: {thread_id: {'metadata': ..., 'messages': [msg_dict, ...]}}
    threads = {}
    # Index: {message_id (int): thread_id (str)}
    msg_index = {}
    # Orphans: {parent_message_id (int): [child_msg_dict, ...]}
    orphans = {}

    # Sort to minimize orphans (parents first)
    pdf_sorted = pdf.sort_values(by="message_id")

    for _, row in pdf_sorted.iterrows():
        try:
            # 1. Clean & Prepare Row
            msg = row.to_dict()
            
            # Clean Spark Artifacts
            for c in ['date_timestamp', '__index_level_0__']:
                if c in msg: del msg[c]

            # Handle Reactions (String -> Python Object)
            raw_rxn = msg.get('reactions')
            if isinstance(raw_rxn, str):
                try:
                    msg['reactions'] = json.loads(raw_rxn)
                except:
                    msg['reactions'] = []
            elif raw_rxn is None:
                msg['reactions'] = []

            # Clean IDs
            msg_id = int(msg['message_id'])
            chat_id = int(msg['chat_id'])
            
            # Handle Parent ID
            parent_id = msg.get('reply_to_message_id')
            if parent_id is not None and parent_id != -1 and not pd.isna(parent_id):
                parent_id = int(parent_id)
            else:
                parent_id = None
            
            msg['reply_to_message_id'] = parent_id # Update dict with clean value

            # --- 2. THREADING LOGIC (Pure Python) ---
            assigned_thread_id = None

            # Scenario A: It is a reply
            if parent_id:
                # Is parent known?
                if parent_id in msg_index:
                    assigned_thread_id = msg_index[parent_id]
                    threads[assigned_thread_id]['messages'].append(msg)
                    msg_index[msg_id] = assigned_thread_id # Register self
                else:
                    # Parent unknown -> Orphan Buffer
                    if parent_id not in orphans:
                        orphans[parent_id] = []
                    orphans[parent_id].append(msg)
                    # We don't index self yet because we don't know our thread ID
            
            # Scenario B: It is a new Root
            else:
                # Create new thread
                new_thread_id = f"{chat_id}_{msg_id}"
                assigned_thread_id = new_thread_id
                
                threads[new_thread_id] = {
                    'messages': [msg],
                    'root_msg': msg
                }
                msg_index[msg_id] = new_thread_id

            # --- 3. ORPHAN RESOLUTION (Did I just save someone?) ---
            # If this message is a parent to waiting orphans, link them now.
            if assigned_thread_id and msg_id in orphans:
                waiting_kids = orphans.pop(msg_id)
                # We use a stack to recursively resolve chains of orphans
                # (e.g. Msg A arrives, unlocking B, which unlocks C)
                stack = [(assigned_thread_id, kid) for kid in waiting_kids]
                
                while stack:
                    t_id, kid_msg = stack.pop()
                    threads[t_id]['messages'].append(kid_msg)
                    
                    kid_id = int(kid_msg['message_id'])
                    msg_index[kid_id] = t_id
                    
                    # Does this kid have orphans waiting for IT?
                    if kid_id in orphans:
                        grandkids = orphans.pop(kid_id)
                        for gk in grandkids:
                            stack.append((t_id, gk))

        except Exception as e:
            # Log error but skip message
            continue

    # --- 4. FORMAT OUTPUT ---
    results = []
    
    # Filter for threads with >= 2 messages
    for t_id, data in threads.items():
        msgs = data['messages']
        if len(msgs) < 2:
            continue
            
        try:
            # Generate Metadata
            root = data['root_msg']
            participants = {m.get('from_id') for m in msgs if m.get('from_id')}
            start_ts = min(int(m['date_unixtime']) for m in msgs)
            end_ts = max(int(m['date_unixtime']) for m in msgs)
            
            # Generate Transcript
            transcript = _generate_transcript(root['chat_name'], msgs)
            
            # Final Serialization (List[Dict] -> JSON String)
            # We use a custom encoder to ensure NumPy types don't sneak in
            def np_encoder(obj):
                if isinstance(obj, (np.integer, np.int64)): return int(obj)
                if isinstance(obj, (np.floating, np.float64)): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return str(obj)

            messages_json = json.dumps(msgs, default=np_encoder)

            results.append({
                'thread_id': str(t_id),
                'chat_id': int(root['chat_id']),
                'chat_name': str(root['chat_name']),
                'week_id': str(root['week_id']),
                'root_message_id': int(root['message_id']),
                'transcript': transcript,
                'messages_json': messages_json,
                'message_count': len(msgs),
                'participant_count': len(participants),
                'start_unixtime': start_ts,
                'end_unixtime': end_ts
            })
        except Exception as e:
            logger.error(f"[Worker] Error finalizing thread {t_id}: {e}")
            continue

    return pd.DataFrame(results)

@step(output_materializers=[SparkDataFrameMaterializer])
def reconstruct_threads(df: DataFrame) -> DataFrame:
    step_name = "reconstruct_threads"
    
    # 1. Validation
    df_silver = enrich_columns(df)
    required = ["chat_id", "message_id", "reply_to_message_id"]
    missing = [c for c in required if c not in df_silver.columns]
    if missing:
        raise ValueError(f"[{step_name}] Missing columns: {missing}")

    # 2. Execution
    df_threads = df_silver.groupBy("chat_id").applyInPandas(
        _worker_thread_engine,
        schema=GOLD_THREAD_SCHEMA
    )

    # 3. Observability
    df_threads.cache()
    thread_count = df_threads.count()

    try:
        step_context = get_step_context()
        step_context.add_output_metadata(
            output_name="output", 
            metadata={"reconstructed_thread_count": thread_count}
        )
    except Exception:
        pass

    logger.info(f"[{step_name}] Success. Reconstructed {thread_count} threads.")
    
    if thread_count == 0:
        logger.warning(f"[{step_name}] WARNING: 0 threads found.")

    return df_threads