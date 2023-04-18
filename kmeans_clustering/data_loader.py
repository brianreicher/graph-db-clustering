import numpy as np
import pickle


class CIFAR10Loader():
    def __init__(self, spark, data_dir) -> None:
        self.spark = spark
        self.data_dir = data_dir

    def _read_batch(self, file_path) -> tuple:
        # helper method to read a batch of CIFAR-10 data from a file
        # takes the path of the file to read as input
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']
        images = images.reshape(-1, 3, 32, 32)
        images = images.transpose((0, 2, 3, 1))
        labels: np.ndarray = np.array(labels)
        return images, labels

    def load(self) -> None:
        # method to load the CIFAR-10 data from the specified directory 
        # TODO: Need to figure out where to load imagebs into
        pass
