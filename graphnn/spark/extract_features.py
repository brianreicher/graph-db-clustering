from pyspark.sql import SparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors, VectorUDT
import cv2
import numpy as np
from ..database.imageDB import ImageDB


class FeatureExtractor():
    def __init__(self, image_dir, batch_size=5) -> None:
        # create a SparkSession for our feature extractor
        self.spark: SparkSession = SparkSession.builder.appName("FeatureExtractor").getOrCreate()

        # define the image directory and batch size
        self.image_dir:str = image_dir
        self.batch_size: int = batch_size

    # define a function to load images in batches and convert them to numpy arrays
    def load_images(self) -> list:
        images = []
        for i in range(self.batch_size):
            # load the image
            img_path = self.image_dir + "/" + batch[i]
            img = cv2.imread(img_path)

            # convert the image to a 1028x1028 numpy array
            img = cv2.resize(img, (1028, 1028))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
        return images
    
    @staticmethod
    def extract_color_histogram(image) -> np.ndarray:
        # Convert the image to the HSV color space
        hsv_image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the ranges for the histogram bins
        hue_range: list[int] = [0, 180]
        saturation_range: list[int] = [0, 256]
        value_range: list[int] = [0, 256]

        # Define the number of bins for each channel
        hue_bins:int = 30
        saturation_bins:int = 32
        value_bins:int = 32

        # Compute the histogram
        histogram:np.ndarray = cv2.calcHist(
            [hsv_image], [0, 1, 2], None,
            [hue_bins, saturation_bins, value_bins],
            [hue_range, saturation_range, value_range]
        )

        # Normalize the histogram
        histogram: np.ndarray = cv2.normalize(histogram, histogram)

        # Reshape the histogram into a 1D array
        histogram: np.ndarray = histogram.reshape(-1)

        return histogram

    # define a function to extract image features
    @staticmethod
    def extract_features(image):
        # perform some feature extraction on the image
        # ...

        # return a list of 10 feature values
        return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # define a function to save image features as a graph in Neo4j
    def save_as_graph(self, batch_features:int) -> None:
        # connect to the Neo4j database
        database: ImageDB = ImageDB("neo4j://localhost:7687", "neo4j", "password")
        database.connect()
        

        # create nodes for each feature
        for feature_idx in range(10):
            nodes = []
            for i in range(self.batch_size):
                nodes.append({
                    "id": i,
                    "value": batch_features[i][feature_idx]
                })

            # create edges between nodes based on some relationship
            edges = []
            for i in range(self.batch_size):
                for j in range(i+1, self.batch_size):
                    edges.append({
                        "from": i,
                        "to": j,
                        "weight": abs(batch_features[i][feature_idx] - batch_features[j][feature_idx])
                    })
            with database.driver.session() as session:
                # save nodes and edges to Neo4j
                session.run("""
                    UNWIND $nodes as node
                    MERGE (n:FeatureNode {id: node.id, value: node.value})
                """, nodes=nodes)
                session.run("""
                    UNWIND $edges as edge
                    MATCH (n1:FeatureNode {id: edge.from})
                    MATCH (n2:FeatureNode {id: edge.to})
                    MERGE (n1)-[:FEATURE_REL {weight: edge.weight}]->(n2)
                """, edges=edges)

                # close the Neo4j session and driver
                session.close()
        database.close()