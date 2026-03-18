from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from typing import List, Dict, Any, Optional, Union
from loguru import logger
import os
from src.configs.settings import settings
from langchain_aws import BedrockEmbeddings
from pymongo.operations import SearchIndexModel


class MongoService:
    """
    A generic service wrapper for MongoDB operations with built-in logging 
    and error handling. Can be instantiated for any collection.
    """
    def __init__(self, collection_name: str, mongo_uri: str = settings.mongo_uri , db_name: str = settings.DB_NAME):
        try:
            self.client = MongoClient(mongo_uri)
            self.db = self.client[db_name]
            self.collection = self.db[collection_name]
            self.collection_name = collection_name
            
            # Lightweight check to ensure connection is alive
            self.client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB collection: '{collection_name}'")
            
        except ConnectionFailure as e:
            logger.critical(f"Failed to connect to MongoDB: {e}")
            raise e

    def find_one(self, query: Dict[str, Any], projection: Optional[Dict] = None) -> Optional[Dict]:
        """Generic find_one with logging."""
        try:
            # Default projection removes _id unless specified otherwise
            proj = projection if projection else {"_id": 0}
            result = self.collection.find_one(query, proj)
            
            if result:
                logger.debug(f"Found document in {self.collection_name} matching: {query}")
            else:
                logger.warning(f"No document found in {self.collection_name} matching: {query}")
                
            return result
        except Exception as e:
            logger.error(f"Error in find_one: {e}")
            return None

    def find_many(self, query: Dict[str, Any] = {}, limit: int = 0, sort_by: str = None, projection: Optional[Dict] = None) -> List[Dict]:
        """Generic find_many with optional sorting and limiting."""
        try:
            proj = projection if projection else {"_id": 0}
            cursor = self.collection.find(query, proj)
            
            if sort_by:
                cursor = cursor.sort(sort_by, -1) # Default to descending
            
            if limit > 0:
                cursor = cursor.limit(limit)
            
            results = list(cursor)
            logger.info(f"Retrieved {len(results)} documents from {self.collection_name}")
            return results
        except Exception as e:
            logger.error(f"Error in find_many: {e}")
            return []

    def insert_one(self, document: Dict) -> Optional[str]:
        """Inserts a single document and returns its ID."""
        try:
            result = self.collection.insert_one(document)
            logger.info(f"Inserted document into {self.collection_name} with ID: {result.inserted_id}")
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
            return None

    def count(self, query: Dict[str, Any] = {}) -> int:
        """Counts documents matching the query."""
        try:
            count = self.collection.count_documents(query)
            logger.info(f"Counted {count} documents in {self.collection_name}")
            return count
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0

    def aggregate(self, pipeline: List[Dict]) -> List[Dict]:
        """Executes a raw aggregation pipeline."""
        try:
            logger.debug(f"Running aggregation on {self.collection_name} with {len(pipeline)} stages")
            results = list(self.collection.aggregate(pipeline))
            logger.info(f"Aggregation returned {len(results)} results")
            return results
        except OperationFailure as e:
            logger.error(f"Aggregation failed: {e}")
            return []

    # --- SAMPLING METHODS ---

    def get_random_sample(self, sample_size: int = 100) -> List[Dict]:
        """
        Fetches a purely random sample using $sample.
        """
        logger.info(f"Fetching random sample of {sample_size} from {self.collection_name}")
        pipeline = [
            {"$sample": {"size": sample_size}},
            {"$project": {"_id": 0}}
        ]
        return self.aggregate(pipeline)

    def get_stratified_sample(self, group_by_field: str, samples_per_group: int = 5) -> List[Dict]:
        """
        Generic Stratified Sampling.
        
        Args:
            group_by_field (str): The field to stratify by (e.g., 'week_id' for threads, 'sender_id' for messages).
            samples_per_group (int): How many items to take from each group.
        """
        logger.info(f"Fetching stratified sample: {samples_per_group} items per '{group_by_field}'")
        
        pipeline = [
            # 1. Filter out documents where the stratification field is missing (optional but safe)
            {"$match": {group_by_field: {"$exists": True, "$ne": None}}},

            # 2. Group buckets
            {"$group": {
                "_id": f"${group_by_field}",
                "items": {"$push": "$$ROOT"}
            }},
            
            # 3. Slice the array to get 'samples_per_group'
            {"$project": {
                "sampled_items": { "$slice": ["$items", samples_per_group] }
            }},
            
            # 4. Flatten back to root level
            {"$unwind": "$sampled_items"},
            {"$replaceRoot": {"newRoot": "$sampled_items"}},
            
            # 5. Clean output
            {"$project": {"_id": 0}}
        ]
        return self.aggregate(pipeline)
    
    def close(self) -> None:
        """Close the MongoDB connection.

        This method should be called when the service is no longer needed
        to properly release resources, unless using the context manager.
        """

        self.client.close()
        logger.debug("Closed MongoDB connection.")
        
class MongoAtlasVectorDB:
    def __init__(self, uri=settings.mongo_uri, db_name=settings.DB_NAME, collection_name="hybrid_summaries", embedding_model_id="amazon.titan-embed-text-v2:0"):
        """Initializes the MongoDB connection and the Amazon Titan V2 embedding model."""
        try:
            self.client = MongoClient(uri)
            self.db = self.client[db_name]
            
            # --- THE FIX: Explicitly create the collection if it's missing ---
            if collection_name not in self.db.list_collection_names():
                logger.info(f"Collection '{collection_name}' not found. Creating it now...")
                self.db.create_collection(collection_name)
                
                # Assign the collection
                self.collection = self.db[collection_name]
                
                # Automatically build the Titan V2 Vector Index
                self._create_vector_index()
            else:
                self.collection = self.db[collection_name]

            # Initialize Amazon Titan V2 Embeddings
            self.embeddings = BedrockEmbeddings(model_id=embedding_model_id)
            
            logger.info(f"Successfully connected to MongoDB Atlas Local: {db_name}.{collection_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def _create_vector_index(self):
        """Programmatically creates the Atlas Vector Search index for Titan V2 embeddings."""
        logger.info("Building Vector Search Index for Amazon Titan V2 (1024 dimensions)...")
        
        search_index_model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "numDimensions": 1024, # Matches Titan V2
                        "similarity": "cosine"
                    },
                    {
                        "type": "filter",
                        "path": "l1_domain"
                    }
                ]
            },
            name="titan_vector_index",
            type="vectorSearch"
        )
        
        try:
            self.collection.create_search_index(model=search_index_model)
            logger.info("Vector Search Index created successfully.")
        except Exception as e:
            logger.error(f"Failed to create Vector Search Index: {e}")

    def embed_and_store(self, thread_id: str, summary: str, l1_domain: str):
        """Generates a Titan V2 vector and upserts the document into MongoDB."""
        try:
            # 1. Generate the vector using Titan V2
            vector = self.embeddings.embed_query(summary)
            
            # 2. Build the Hybrid Document
            document = {
                "thread_id": thread_id,
                "summary": summary,
                "l1_domain": l1_domain,
                "embedding": vector  # Store the 1024-dimensional float array
            }
            
            # 3. Upsert into MongoDB
            self.collection.update_one(
                {"thread_id": thread_id}, 
                {"$set": document}, 
                upsert=True
            )
            logger.info(f"Successfully embedded and stored thread {thread_id}")
            
        except Exception as e:
            logger.error(f"Failed to embed and store thread {thread_id}: {e}")
            
            
    def vector_search(self, embedding: list, top_k: int = 3, l1_domain: str = None) -> list:
        """Runs a vector search for the top K closest summaries based on a Titan embedding."""
        
        vector_search_stage = {
            "$vectorSearch": {
                "index": "titan_vector_index",
                "queryVector": embedding,
                "path": "embedding",
                "numCandidates": top_k * 10,  # Best practice: 10-20x the limit
                "limit": top_k,
            }
        }
        
        # Apply pre-filtering inside the vector search stage
        if l1_domain:
            vector_search_stage["$vectorSearch"]["filter"] = {"l1_domain": l1_domain}
            
        pipeline = [
            vector_search_stage,
            {
                "$project": {
                    "_id": 0, 
                    "thread_id": 1, 
                    "summary": 1, 
                    "l1_domain": 1, 
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        try:
            return list(self.collection.aggregate(pipeline))
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []