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
                        MATCH (n)-[e]-(m)
                        WHERE n.graph_name = "{self.name}"
                        RETURN id(n) as source, id(m) as target, n.f1 as f1, n.f2 as f2, n.f3 as f3, n.f4 as f4, n.f5 as f5, e.weight as weight
                    """

        # Execute the query and extract the nodes and edges
        with self.database.driver.session() as session:
            result = session.run(query)
            nodes:set = set()
            edges:list = []
            for record in result.records():
                source = record["source"]
                target = record["target"]
                f1 = record["f1"]
                f2 = record["f2"]
                f3 = record["f3"]
                f4 = record["f4"]
                f5 = record["f5"]
                weight = record["weight"]
                nodes.add(source)
                nodes.add(target)
                edges.append((source, target, weight, f1, f2, f3, f4, f5))

        # Convert the nodes and edges to a PyTorch Geometric graph
        edge_index: torch.Tensor = torch.tensor(edges)[:, :2].t().contiguous()
        edge_attr: torch.Tensor = torch.tensor(edges)[:, 2:].float()
        x: torch.Torch = torch.tensor(list(nodes)).unsqueeze(1).float()
        data:Data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
