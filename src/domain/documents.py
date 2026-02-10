import os
import pandas as pd
import json
import re
import logging
import numpy as np

from typing import Optional, List, Dict, Any, Tuple, Set, Union
from datetime import datetime
from collections import Counter
from pydantic import Field, BaseModel, ValidationError,  ConfigDict
from typing_extensions import Annotated
from zenml import step, ArtifactConfig


# In test.ipynb
import os
import sys

# Adds the current directory (tel_repo) to Python's path
sys.path.append(os.getcwd())

#import llm_engineering

#os.chdir("/mnt/c/Users/12404/Downloads/ai engineering/dev-projects/tel_repo")

class Reaction(BaseModel):
    """
    Sub-model for handling reaction details if present.
    Adjust fields based on your specific JSON structure (usually 'emoji' and 'count').
    """
    emoji: Optional[str] = None
    count: Optional[int] = None
    # Allow extra fields since reaction structures can vary by export version
    model_config = {"extra": "ignore"}

class TelegramChatDocument(BaseModel):
    id : str
    message_id: int
    chat_id : int
    chat_name:str
    date: str    
    date_unixtime: str  # Telegram exports often store this as a string
    week_id : str
    
    # 'from' is a reserved Python keyword, so we use 'sender' and map it
    sender: str
    from_id: int  # IDs in exports are often strings (e.g., "user123456")
    
    # Text can be a plain string or a list of entities (mixed strings and dicts)
    text: Union[str, List[Any]] = ""
    
    # Reactions are often a list of dictionaries
    reactions: Optional[List[Reaction]] = []
    
    # --- NEW FIELDS FOR THREADING ---
    # The ID of the message this message is replying to (nullable for root messages)
    reply_to_message_id: Optional[int] = None
    
    # --- REQUIRED SETTING ---
    class Settings:
        name = 'TelegramChatDocument'
        
    # @field_validator('text', mode='before')
    # @classmethod
    # def flatten(cls, v):
    #     return str(v)


class TelegramUserDocument(BaseModel):
    telegram_id: int = Field(..., description="Unique ID from Telegram (from_id)")
    display_name: Optional[str] = None
    is_bot: bool = False

    # --- REQUIRED SETTING ---
    class Settings:
        name = "user_telegram"
        
class UserTracker(BaseModel):
    """
    Helper class to track unique user IDs during processing.
    """
    # Use a SET to efficiently track IDs already seen (O(1) lookup)
    unique_user_ids: Set[int] = Field(default_factory=set)#set()
    # Store validated Pydantic models for a single, final write
    user_documents: List[TelegramUserDocument] = Field(default_factory=list)
    
    
class ChatMessageTracker(BaseModel):
    """
    Helper class to track unique chat message IDs during processing.
    """
    # Use a SET to efficiently track IDs already seen (O(1) lookup)
    unique_message_ids: Set[str] = Field(default_factory=set)
    # Store validated Pydantic models for a single, final write
    chat_message_documents: List[TelegramChatDocument] = Field(default_factory=list)
    
   
    
class ThreadReactionSummary(BaseModel):
    total_count: int = 0
    unique_emojis: Set[str] = Field(default_factory=set)
    # New: Track specific counts for sentiment analysis (e.g., {'🔥': 10, '😡': 2})
    emoji_counts: Counter[str] = Field(default_factory=Counter)

    def update(self, reactions: List[Reaction]):
        for r in reactions:
            self.total_count += r.count
            self.unique_emojis.add(r.emoji)
            self.emoji_counts[r.emoji] += r.count
    

    


class ThreadDocument(BaseModel):
    """
    Optimized for RAG & Virality Analysis.
    Captures the Narrative (Messages), Actors (Participants), and Pulse (Reactions).
    """
    thread_id: str
    chat_id: int
    chat_name: str
    root_message_id: int
    week_id : str
    
    # --- NEW FIELD: The Concatenated String ---
    # This stores the final rendered text for RAG indexing
    transcript: Optional[str] = Field(None, description="The full, chronological conversation script formatted for LLMs.")
    
    # Core Data
    messages: List[TelegramChatDocument] = Field(default_factory=list)
    
    # Metadata for Analysis
    participant_ids: Set[int] = Field(default_factory=set)
    # Map ID to Display Name for resolving "He said/She said" in summaries
    participant_names: Dict[int, str] = Field(default_factory=dict)
    
    reaction_summary: ThreadReactionSummary = Field(default_factory=ThreadReactionSummary)
    
    # Time window tracking for "Velocity" calculation
    start_unixtime: float = Field(default=float('inf'))
    end_unixtime: float = Field(default=float('-inf'))

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_message(self, msg: TelegramChatDocument):
        """
        Efficiently updates state without full re-validation.
        """
        self.messages.append(msg)
        
        # 1. Update Participants (Identity)
        if msg.from_id:
            self.participant_ids.add(msg.from_id)
            if msg.sender:
                self.participant_names[msg.from_id] = msg.sender
        
        # 2. Update Reactions (Sentiment)
        if msg.reactions:
            self.reaction_summary.update(msg.reactions)
            
        # 3. Update Time Window (Velocity)
        try:
            ts = float(msg.date_unixtime)
            if ts < self.start_unixtime: self.start_unixtime = ts
            if ts > self.end_unixtime: self.end_unixtime = ts
        except (ValueError, TypeError):
            pass

    def render_for_llm(self) -> str:
        """
        Generates the 'Script' for the AI Summarizer.
        Outputs a dense, token-efficient text representation.
        """
        # 1. Header with Metadata Signals
        duration_min = (self.end_unixtime - self.start_unixtime) / 60
        velocity = len(self.messages) / duration_min if duration_min > 0 else len(self.messages)
        
        top_emojis = self.reaction_summary.emoji_counts.most_common(3)
        sentiment_signal = " ".join([f"{e}({c})" for e, c in top_emojis])
        
        lines = [
            f"### CONVERSATION THREAD: {self.chat_name} ",
            f"### METADATA: {len(self.messages)} msgs | {len(self.participant_ids)} participants | {int(duration_min)} mins duration | thread_id: {self.thread_id}",
            f"### SENTIMENT/REACTIONS: {sentiment_signal}" if sentiment_signal else "### SENTIMENT: Neutral ",
            "### TRANSCRIPT: ",
        ]

        # 2. Sort messages to ensure chronological narrative
        # (We sort here on render, rather than on every insert, for speed)
        sorted_msgs = sorted(self.messages, key=lambda x: float(x.date_unixtime))

        for msg in sorted_msgs:
            # Format: [HH:MM] Name: Text {🔥5}
            
            ts_str = datetime.fromtimestamp(int(float(msg.date_unixtime))).strftime('%Y-%m-%d-%H-%M-%S')
            
            # reaction string (token efficient)
            reacts = ""
            if msg.reactions:
                # Get top 2 reactions for this specific message to save tokens
                top = sorted(msg.reactions, key=lambda x: x.count, reverse=True)[:2]
                reacts = f" {{rxn: {' '.join([f'{r.emoji}{r.count}' for r in top])}}}"
            
            lines.append(f"[{ts_str}] {msg.sender}: {msg.text}{reacts}") #lines.append(f"[{ts_str}] {msg.sender or 'Unknown'}: {msg.text}{reacts}")
            
        return "\n".join(lines)

# 2. The Processor (ThreadTracker)
# This class remains the "Engine." I have cleaned up the logic to ensure strictly O(1) handling of message linking.

class ThreadTracker:
    """
    Ingests a chaotic stream of Telegram messages and stitches them into
    coherent narratives using the ThreadDocument model.
    """
    def __init__(self):
        # The Database of Threads
        self.threads: Dict[str, ThreadDocument] = {}
        
        # The Index: Maps ANY message_id (composite) -> thread_id
        # Allows O(1) retrieval of the thread for any reply.
        self.msg_to_thread_index: Dict[str, str] = {}
        
        # The Waiting Room: Maps parent_message_id (int) -> List[Messages]
        # Stores messages that arrived before their parent.
        self.orphans: Dict[int, List[TelegramChatDocument]] = {}

    def process_item(self, msg: TelegramChatDocument):
        """
        Main ingestion method. Call this for every message in the JSON stream.
        """
        parent_int_id = msg.reply_to_message_id
        parent_comp_id = f"{msg.chat_id}_{parent_int_id}" if parent_int_id else None

        # --- SCENARIO 1: IT IS A REPLY ---
        if parent_int_id:
            # Do we already know which thread the PARENT belongs to?
            thread_id = self.msg_to_thread_index.get(parent_comp_id)

            if thread_id:
                # Success: Parent is known. Link immediately.
                self._link_message_to_thread(thread_id, msg)
            else:
                # Failure: Parent unknown (Orphan). Buffer it.
                if parent_int_id not in self.orphans:
                    self.orphans[parent_int_id] = []
                self.orphans[parent_int_id].append(msg)

        # --- SCENARIO 2: IT IS A NEW ROOT (No Reply ID) ---
        else:
            # Create a new thread container
            new_thread_id = str(f"{msg.chat_id}_{str(int(msg.message_id))}")
            new_thread = ThreadDocument(
                thread_id=new_thread_id,
                week_id=msg.week_id,
                chat_id=msg.chat_id,
                chat_name=msg.chat_name,
                root_message_id=msg.message_id
            )
            
            # Register the thread
            self.threads[new_thread_id] = new_thread
            
            # Link this root message (it also checks for its own orphans!)
            self._link_message_to_thread(new_thread_id, msg)

    def _link_message_to_thread(self, thread_id: str, msg: TelegramChatDocument):
        """
        Helper: Adds message to thread, updates index, and resolves orphans.
        """
        # 1. Add to the actual Document
        self.threads[thread_id].add_message(msg)
        
        # 2. Update Index (So future replies to THIS message find the thread)
        self.msg_to_thread_index[msg.id] = thread_id
        
        # 3. Check Orphan Buffer (Did anyone reply to this message before it arrived?)
        # We look up by the integer ID because that's how Telegram stores reply refs
        if msg.message_id in self.orphans:
            waiting_children = self.orphans.pop(msg.message_id)
            
            # Recursively link the waiting children
            for child in waiting_children:
                self._link_message_to_thread(thread_id, child)