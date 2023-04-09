import torch
from torch_geometric.data import Dataset
from .torch_graph import TorchImageGraph

class ImageGraphDataset(Dataset):
    def __init__(self, root, name, transform=None, pre_transform=None) -> None:
        self.name = name
        self.graphs:list = []
        super(ImageGraphDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self) -> list:
        return []

    @property
    def processed_file_names(self) -> list:
        return [f"{self.name}.pt"]

    def download(self) -> None:
        pass

    def process(self) -> None:
        # Load the geometric objects created by TorchImageGraph
        image_graph: TorchImageGraph = TorchImageGraph(self.name)
        image_graph.extract_graph()
        self.graphs.append(image_graph.graph)

        # Save the dataset to a file
        data, slices = self.collate(self.graphs)
        torch.save((data, slices), self.processed_paths[0])

    def len(self) -> int:
        return len(self.graphs)

    def get(self, idx):
        data, slices = torch.load(self.processed_paths[idx])
        return data
