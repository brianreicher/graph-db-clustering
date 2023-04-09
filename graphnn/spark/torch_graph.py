import torch
from torch_geometric.data import Data
from ..database.imageDB import ImageDB

class TorchImageGraph():

    def __init__(self, name) -> None:
       # connect to the Neo4j database
        self.database: ImageDB = ImageDB("neo4j://localhost:7687", "neo4j", "password")
        self.database.connect()
        self.name = name
        self.graph:Data = None
    
    def extract_graph(self) -> None:
        query:str = f"""
                        MATCH (n:Image)-[]-(m)
                        WHERE n.name = "{self.name}"
                        RETURN id(n) as source, id(m) as target
                    """

        with self.database.driver.session() as session:
            result = session.run(query)
            edges = [(record["source"], record["target"]) for record in result.records()]

            # Convert the edges to PyTorch Geometric format
            edge_index: torch.Tensor = torch.tensor(edges).t().contiguous()

            # Assign a PyTorch Geometric Data object
            self.graph: Data =Data(edge_index=edge_index)