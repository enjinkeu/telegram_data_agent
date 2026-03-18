from neo4j import GraphDatabase
import logging
import re
from typing import List, Dict
from src.configs.settings import settings

logger = logging.getLogger(__name__)

class Neo4jConnector:
    def __init__(self, uri=settings.NEON4J_URI, user=settings.NEON4J_USER, password=settings.NEON4J_PASS):
        """Initializes the connection to the Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Verify connectivity
            self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Closes the database connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed.")

    def _sanitize_predicate(self, predicate: str) -> str:
        """
        Neo4j relationship types cannot be parameterized like standard variables.
        They must be injected into the query string safely. 
        This ensures the LLM's output is safe, uppercase, and snake_case.
        """
        clean_str = re.sub(r'[^a-zA-Z0-9_]', '_', predicate)
        return clean_str.upper()

    def ingest_triplets(self, thread_id: str, topic: str, triplets: List[Dict]):
        """Ingests a list of S-P-O dictionaries into the Graph Database."""
        with self.driver.session() as session:
            for triplet in triplets:
                subject = triplet.get("subject", "").strip()
                predicate = triplet.get("predicate", "").strip()
                object_node = triplet.get("object", "").strip()
                context = triplet.get("context", "").strip() # Extract context

                if not subject or not predicate or not object_node:
                    continue

                rel_type = self._sanitize_predicate(predicate)

                query = f"""
                MERGE (s:Entity {{name: $subject}})
                MERGE (o:Entity {{name: $object_node}})
                MERGE (s)-[r:`{rel_type}`]->(o)
                
                // Store provenance AND context on the relationship edge
                SET r.thread_id = $thread_id,
                    r.topic = $topic,
                    r.context = $context
                """
                
                try:
                    session.run(
                        query, 
                        subject=subject, 
                        object_node=object_node, 
                        thread_id=thread_id, 
                        topic=topic,
                        context=context # Pass parameter
                    )
                except Exception as e:
                    logger.error(f"Failed to ingest triplet ({subject} -> {rel_type} -> {object_node}): {e}")
                    
                    
    def fetch_triplets_by_threads(self, thread_ids: List[str]) -> List[Dict]:
        """Fetches all triplets and their context for a list of thread_ids."""
        if not thread_ids:
            return []
            
        query = """
        MATCH (s:Entity)-[r]->(o:Entity)
        WHERE r.thread_id IN $thread_ids
        RETURN s.name AS subject, TYPE(r) AS predicate, o.name AS object, r.context AS context, r.thread_id AS thread_id
        """
        results = []
        with self.driver.session() as session:
            try:
                for record in session.run(query, thread_ids=thread_ids):
                    results.append({
                        "subject": record["subject"],
                        "predicate": record["predicate"].lower(),
                        "object": record["object"],
                        "context": record.get("context"),
                        "thread_id": record.get("thread_id")
                    })
            except Exception as e:
                logger.error(f"Failed to fetch triplets for threads {thread_ids}: {e}")
        return results