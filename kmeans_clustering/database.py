import neo4j
from neo4j import GraphDatabase


class ImageDB():
    """
    Neo4j graph database to contain image embeddings for KNN and KMC classification.
    Attributes:
        uri (str): The URI of the Neo4j server.
        user (str): The username for the Neo4j server.
        password (str): The password for the Neo4j server.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        """
        Constructs a new ImageDB object.
        Args:
            uri (str): The URI of the Neo4j server.
            user (str): The username for the Neo4j server.
            password (str): The password for the Neo4j server.
        """
        self.uri: str = uri
        self.user: str = user
        self.password: str = password
        self.driver: GraphDatabase.driver = None

    def connect(self) -> None:
        """
        Connects to the Neo4j server.
        """
        try:
            self.driver: GraphDatabase.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        except neo4j.exceptions.ServiceUnavailable as e:
            print(f"Failed to connect to Neo4j server: {e}")

    def disconnect(self) -> None:
        """
        Disconnects from the Neo4j server.
        """
        self.driver.close()    

    def flush_database(self) -> None:
        """
        Deletes all nodes and edges from the graph database.
        """
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print(f"Deleted all nodes and edges")


    def create_relationship(self, start_node_id:str, end_node_id:str, relationship_type:str, props:str) -> None:
        """
        Creates a new relationship between two nodes given their IDs and a relationship type.
        Args:
            start_node_id (str): The ID of the start node.
            end_node_id (str): The ID of the end node.
            relationship_type (str): The type of relationship to create.
            props (str): The properties of the relationship.
        """
        with self.driver.session() as session:
            # beings transaction
            tx = session.begin_transaction()
            # query string
            query: str = f"MATCH (a),(b) WHERE a.name={start_node_id} AND b.name={end_node_id} CREATE (a)-[r:{relationship_type} {props}]->(b)"
            # executes and commits
            tx.run(query)
            tx.commit()