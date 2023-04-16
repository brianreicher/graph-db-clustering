import cv2
import numpy as np
import pyspark
from tqdm import tqdm
import os
import neo4j
from neo4j import GraphDatabase
from pyspark.sql import SparkSession
from PIL import Image, ImageFilter
import mahotas
import pickle

class CIFAR10Loader():
    def __init__(self, spark, data_dir):
        self.spark = spark
        self.data_dir = data_dir

    def _read_batch(self, file_path):
        # helper method to read a batch of CIFAR-10 data from a file
        # takes the path of the file to read as input
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        images = data_dict[b'data']
        labels = data_dict[b'labels']
        images = images.reshape(-1, 3, 32, 32)
        images = images.transpose((0, 2, 3, 1))
        labels = np.array(labels)
        return images, labels

    def load(self):
        # method to load the CIFAR-10 data from the specified directory 
        # TODO: Need to figure out where to load images into
        pass


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


class FeatureExtractor():
    def __init__(self, image_dir, batch_size=5, epochs=100) -> None:
        # create a SparkSession for our feature extractor
        self.spark: SparkSession = SparkSession.builder.appName("FeatureExtractor").getOrCreate()

        # define the batch size
        self.batch_size: int = batch_size

        self.batch_index:int = 0
        self.batch = None
        self.epochs:int = epochs
        # connect to the Neo4j database
        self.database: ImageDB = ImageDB("neo4j://localhost:7687", "neo4j", "password")
        self.database.connect()

        self.image_dir:str = image_dir
        self.file_list:list = []

        # Iterate over all files in the directory
        for filename in os.listdir(image_dir):

            filepath: str = os.path.join(image_dir, filename)
            # Check if the file is a regular file (i.e., not a directory)
            if os.path.isfile(filepath):
                # Add the file path to the list
                self.file_list.append(filepath)

     # define a function to load images in batches and convert them to numpy arrays using spark 
    def load_images(self) -> None:
        images_rdd = self.spark.sparkContext.parallelize(self.file_list)

        def load_image_np(img_path):
            try:
                # load the image
                img = cv2.imread(img_path)

                # convert the image to a 32x32 numpy array
                img: np.ndarray = cv2.resize(img, (32, 32))
                img: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                return (img_path, img)
            except:
                return (img_path, None)

        images_rdd = images_rdd.map(load_image_np).filter(lambda x: x[1] is not None)

        images = images_rdd.collectAsMap()
        self.batch: dict = images
        self.batch_index += len(images)


    @staticmethod
    def extract_color_histogram(image) -> np.ndarray:

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
            [image], [0, 1, 2], None,
            [hue_bins, saturation_bins, value_bins],
            [hue_range, saturation_range, value_range]
        )

        # Normalize the histogram
        histogram: np.ndarray = cv2.normalize(histogram, histogram)

        # Reshape the histogram into a 1D array
        histogram: np.ndarray = histogram.reshape(-1)

        return histogram

    @staticmethod
    def extract_features(img: np.ndarray) -> list:
        # Extract statistical features: TODO: add more
        return [np.mean(img), np.std(img), np.median(img), np.min(img), np.max(img), np.linalg.det(np.corrcoef(img)), np.linalg.det(np.cov(img))]

    def insertImageGraph(self) -> None:
        if self.batch is None:
            self.load_images()

        for label, img in self.batch.items():
            label: str = label[label.index('/'):]
            print(label)
            feat_array: list = self.extract_features(img)
            print(feat_array)
            contour_array: np.ndarray = self.get_contour_features(img)
            edge_array:np.ndarray= self.detect_edges(img)

            feat_array.append(np.mean(contour_array))
            feat_array.append(np.mean(edge_array))

            with self.database.driver.session() as session:
                # Create the image node
                session.write_transaction(
                    lambda tx: tx.run("CREATE (:Image {name: $name, mean: $mean, std: $std, median: $median, min: $min, max: $max, corrcoef: $corrcoef, covariance: $cov, centroid: false})", name=label, mean=feat_array[0], std=feat_array[1], median=feat_array[2], min=feat_array[3], max=feat_array[4], corrcoef=feat_array[5], cov=feat_array[6])
                )

    def initCentroids(self, k=2) -> None:
        query:str = f"""MATCH (n:Image)
                       WITH n, rand() as r
                       ORDER BY r
                       LIMIT {k}
                       SET n.centroid = true
                    """
        with self.database.driver.session() as session:
            session.run(query)
    
    @staticmethod
    def get_contour_features(img) -> list:
        # Convert image to grayscale and apply binary threshold
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate contour features for each contour
        contour_features = []
        for contour in contours:
            # Calculate perimeter, area, and solidity
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            hull_area = cv2.convexHull(contour, returnPoints=False)
            if hull_area.all() > 0:
                solidity = area / float(hull_area)
            else:
                solidity = 0

            # Calculate extent, equivalent diameter, and orientation
            _, _, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            if rect_area > 0:
                extent = float(area) / rect_area
                equivalent_diameter = np.sqrt(4 * area / np.pi)
                # _, _, angle = cv2.fitEllipse(contour)
                angle: float = 3.14/2 # TODO: UPDATE
            else:
                extent = 0
                equivalent_diameter = 0
                angle = 0

            # Append contour features to list
            contour_features.append([perimeter, area, solidity, extent, equivalent_diameter, angle])

        return contour_features
    
    @staticmethod
    def detect_edges(image_array) -> np.array:
        # Convert numpy array to Pillow Image object
        image: Image = Image.fromarray(image_array)

        # Convert image to grayscale
        image = image.convert('L')

        # Apply Canny edge detection algorithm
        edges = image.filter(ImageFilter.FIND_EDGES)

        # Convert edges back to numpy array and return
        return np.array(edges)
    
    # Extracts texture features from an image array using the Haralick texture descriptor.
    def extract_texture_features(image: np.ndarray) -> np.ndarray:

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate the Haralick texture features
        features = mahotas.features.haralick(gray)

        # Flatten the feature matrix into a 1D feature vector
        features = features.flatten()

        # A 13-element feature vector representing the texture features of the image.
        return features
    
    
    def heursitic(self) -> None:         
    # calculate and assign nodes
        query: str = """
                        MATCH (n:Image {centroid: false}), (c:Image {centroid: true})
                        WITH n, c, abs(n.mean - c.mean) AS difference
                        ORDER BY difference ASC
                        WITH n, collect({centroid: c, difference: difference})[0] AS closest 
                        WITH closest.centroid AS cent, closest.difference as diff
                        CREATE (n)-[:CLOSEST_TO {difference: diff}]->(cent)
                     """
        with self.database.driver.session() as session:
            session.run(query)
    
    def recalcCentroid(self) -> None:
        # re-evaluate centroids
        query:str = """
                        MATCH (centroid:Image {centroid: true})<-[:BELONGS_TO]-(n:Image)
                        WITH centroid, avg(n.mean) AS meanFeature1, avg(n.std) AS meanFeature2
                        CREATE (:Image {name: newCluster, mean: meanFeature1, std: meanFeature2, centroid: true})
                        MATCH (c:Image {centroid: true})
                        SET c.centroid = false
                        MATCH ()-[r]-()
                        DELETE r
                    """
        with self.database.driver.session() as session:
            session.run(query)

    def cosine_similarity(self) -> None:
        # finds the cosine similarity between 2 given nodes (also uses graph-algorithms library from cypher)
        # TODO: needs to specify a feature(s) from a node 
        query: str = """
                        MATCH (a:Image {id: $node1_id})-[:HAS_FEATURE]->(af:Feature)
                        MATCH (b:Image {id: $node2_id})-[:HAS_FEATURE]->(bf:Feature)
                        WITH a, b, collect(af.value) AS a_features, collect(bf.value) AS b_features
                        WITH a, b, algo.similarity.cosine(a_features, b_features) AS cosine_sim
                        RETURN cosine_sim
                    """
        with self.database.driver.session() as session:
            session.run(query)
    
    def train(self) -> None:
        for _ in range(self.epochs):
            self.heursitic()
            self.recalcCentroid()
            print(_)


if __name__ == "__main__":
    print("initialize driver")
    fe: FeatureExtractor = FeatureExtractor(image_dir='../data', batch_size=1)
    fe.load_images()
    fe.database.flush_database()

    fe.insertImageGraph()
    fe.initCentroids()
    fe.train()
    fe.spark.stop()
