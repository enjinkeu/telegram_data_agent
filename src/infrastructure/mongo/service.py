from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from typing import List, Dict, Any, Optional, Union
from loguru import logger
import os
from src.configs.settings import settings


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