import logging
import pandas as pd
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

class PersistenceService:

    def __init__(self):
        self.uri = "bolt://localhost:7687"
        self.user = "neo4j"
        self.password = "4EkUAcbMLQy8yB3"
        self.dbName = "architectdb"

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        except Exception as e:
            print("Failed to create the driver:", e)

    def close(self):
        if self.driver is not None:
            self.driver.close()
    
    def query(self, query, db=None):
        assert self.driver is not None, "Driver not initialized!"

        session = None
        response = None
        try: 
            session = self.__driver.session(database=db) if db is not None else self.driver.session() 
            response = list(session.run(query))
        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        return response

    # Function to create and update nodes and relationships in Neo4j
    @staticmethod
    def _create_and_update_graph(tx, entity1, relation, entity2):
        
        relation = relation.replace(" ", "_")
        # convert to lowercase
        entity1 = entity1.lower()
        entity2 = entity2.lower()
        relation = relation.lower()

        query = f"""
        MERGE (e1:Entity {{name: $entity1}})
        MERGE (e2:Entity {{name: $entity2}})
        MERGE (e1)-[r:{relation}]->(e2)
        """
        tx.run(query, entity1=entity1, entity2=entity2)

    def update_graph(self):
        csv_file = 'data/documents/relations_data_graph_v2.csv'
        data = pd.read_csv(csv_file)
        with self.driver.session(database=self.dbName) as session:
            for _, row in data.iterrows():
              entity1 = row['Entity1']
              relation = row['Relationship']
              entity2 = row['Entity2']
              print(f"Updating graph with {entity1} {relation} {entity2}")

              session.execute_write(self._create_and_update_graph, entity1=entity1, relation=relation, entity2=entity2)
        print("Graph database updated successfully.")
    
    def find_relationships_by_entity(self, entity_name):
        query = """
            MATCH (e1 {name: $entity_name})-[r]-(e2)
            RETURN e1.name, type(r), e2.name
            UNION
            MATCH (e1)-[r]-(e2 {name: $entity_name})
            RETURN e1.name, type(r), e2.name
        """
        with self.driver.session(database=self.dbName) as session:
            results = session.run(query, entity_name=entity_name)
            relationships = [{"entity1": r[0], "relationship": r[1], "entity2": r[2]} for r in results]
            return relationships
    
    def find_possible_relationships_by_entity(self, entity_name):
        query = """
            MATCH (e1 {name: $entity_name})-[r]->(e2)
            RETURN e1.name, type(r), e2.name
        """
        with self.driver.session(database=self.dbName) as session:
            results = session.run(query, entity_name=entity_name)
            relationships = [{"entity1": r[0], "relationship": r[1], "entity2": r[2]} for r in results]
            return relationships
    
    def find_possible_relationships_by_entity_and_relation(self, entity_name, relationship_type):
        query = """
            MATCH (e1 {name: $entity_name})-[r]->(e2)
            WHERE type(r) = $relationship_type
            RETURN e1.name, type(r), e2.name
        """
        with self.driver.session(database=self.dbName) as session:
            results = session.run(query, entity_name=entity_name, relationship_type=relationship_type)
            relationships = [{"entity1": r[0], "relationship": r[1], "entity2": r[2]} for r in results]
            return relationships


if __name__ == "__main__":
    # Aura queries use an encrypted connection using the "neo4j+s" URI scheme
    

    # Create the driver, session, and run the query
    conn = PersistenceService()

    # Create Database & Update Graph
    # conn.query("CREATE OR REPLACE DATABASE architectdb")
    # conn.update_graph()

    # print(conn.find_relationships_by_entity("application"))
    # print(conn.find_possible_relationships_by_entity("application"))
    # print(conn.find_possible_relationships_by_entity_and_relation("authentication", "uses"))
    conn.close()