from neo4j import GraphDatabase
import logging
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

class Neo4jConnector:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="llm_engineering"):
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
        """
        Ingests a list of S-P-O dictionaries into the Graph Database.
        """
        # We use a context manager for the session to ensure it closes safely
        with self.driver.session() as session:
            for triplet in triplets:
                subject = triplet.get("subject", "").strip()
                predicate = triplet.get("predicate", "").strip()
                object_node = triplet.get("object", "").strip()

                # Skip empty extractions
                if not subject or not predicate or not object_node:
                    continue

                rel_type = self._sanitize_predicate(predicate)

                # The Cypher Query
                # We tag all nodes with a generic :Entity label for easy querying later
                query = f"""
                MERGE (s:Entity {{name: $subject}})
                MERGE (o:Entity {{name: $object_node}})
                MERGE (s)-[r:`{rel_type}`]->(o)
                
                // We store the provenance (where this claim came from) directly on the relationship edge
                SET r.thread_id = $thread_id,
                    r.topic = $topic
                """
                
                try:
                    session.run(
                        query, 
                        subject=subject, 
                        object_node=object_node, 
                        thread_id=thread_id, 
                        topic=topic
                    )
                except Exception as e:
                    logger.error(f"Failed to ingest triplet ({subject} -> {rel_type} -> {object_node}): {e}")