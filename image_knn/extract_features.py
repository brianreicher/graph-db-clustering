import cv2
import numpy as np
import pyspark
from tqdm import tqdm
import os
import neo4j
from neo4j import GraphDatabase
from pyspark.sql import SparkSession


class ImageDB():
    """
    Neo4j graph database to contain image embeddings for GNN classification.
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
    def __init__(self, image_dir, batch_size=5) -> None:
        # create a SparkSession for our feature extractor
        self.spark: SparkSession = SparkSession.builder.appName("FeatureExtractor").getOrCreate()

        # define the batch size
        self.batch_size: int = batch_size

        self.batch_index:int = 0
        self.batch = None

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

    # define a function to load images in batches and convert them to numpy arrays
    def load_images(self) -> None:
        images:dict = {}
        for img_path in tqdm(self.file_list):
            # load the image
            img = cv2.imread(img_path)

            # convert the image to a 1028x1028 numpy array
            img: np.ndarray = cv2.resize(img, (1028, 1028))
            img: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images[img_path] = img
            self.batch_index += 1
        print(images)
        self.batch: dict = images
        
    def kill_session(self) -> None:
        self.spark.stop()

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

    # define a function to extract image features
    @staticmethod
    def extract_features(img: np.ndarray) -> list:
        # Extract statistical features: TODO: add more
        return [np.mean(img), np.std(img), np.median(img), np.min(img), np.max(img)]

    def vectorAssemblerFeatures(self, img: np.ndarray):
        # Apply a Gaussian blur to the grayscale image
        blur_image:np.ndarray = cv2.GaussianBlur(img, (5, 5), 0)

        # Apply Canny edge detection to the blurred image
        edges_image: np.ndarray = cv2.Canny(blur_image, 100, 200)

        # Flatten the image arrays into a single vector
        vector_assembler: VectorAssembler = VectorAssembler(inputCols=["edges"], outputCol="features")
        data: list = [(edges_image.ravel(),)]
        df:pyspark.DataFrame = self.spark.createDataFrame(data, ["edges"])
        df = vector_assembler.transform(df)

        # Display the resulting features
        features = df.select("features").collect()[0][0]
        return features

    def insertImageGraph(self) -> None:
        if self.batch is None:
            self.load_images()

        for label, img in self.batch.items():
            feat_array: list = self.extract_features(img)
            feat_array.append(np.corrcoef(feat_array))
            print(feat_array)

            with self.database.driver.session() as session:
                # Create the image node
                session.write_transaction(
                    lambda tx: tx.run("CREATE (:Image {name: $name, features: $feature_vec})", name=label, feature_vec=feat_array)
                )

    def eucliean_distance(self, node1: dict, node2: dict) -> float:
        """
        Calculates the similarity score between two images based on their attribute values.
        Args:
            nod1 (dict): The first image.
            node2 (dict): The second image.
        Returns:
            float: The similarity score between the two images.
        """

        def process_dict(d) -> np.ndarray:
            """
            Processes input dictionaries to scrape numerical and boolean values to add to sim score vector.
            """
            values:list = []
            for key, value in d.items():
                if key not in self.exclude_keys: # ignore element if it is in the ignore list
                    if isinstance(value, bool): # map booleans to 0/1
                        values.append(int(value))
                    elif isinstance(value, (int, float)): # extract floats/ints
                        values.append(value)

            # Convert the list to a Numpy array
            return np.array(values)
        
        # convert the two dicts
        print(node1, node2)

        # Calculate Euclidean distance between the tracks
        return np.linalg.norm((node1-node2))

    def evaluate_metrics(self, threshold=0) -> None:
        """
        Method for evaluating a given metric threshold over a random batch of nodes.
        Args:
            threshold (float): The threshold to evaluate whether a relationship should be created .
        """
        with self.database.driver.session() as session:
            for image in tqdm(self.file_list): 
                # create a dict of the track props
                node1_values:dict = session.run(f"MATCH (i:Image) WHERE i.name = {image} RETURN i").single()['i']._properties
                for pair_image in self.file_list:
                    # create a dictionary of the random props
                    if pair_image != image:
                        node2_values:dict = session.run(f"MATCH (i:Image) WHERE i.name = {pair_image} RETURN i").single()['i']._properties
                    else:
                        continue
                    # eval the sim score
                    similarity_score: float = self.eucliean_distance(node1_values, node2_values)

                    # create a relationship if the sim score is above the threshold
                    if similarity_score > threshold:
                        self.database.create_relationship(image, pair_image, "MATCHED", f"{{sim_score: {similarity_score}}}")

    def predict(self, name) -> None:
        with self.database.driver.session() as session:
            result = session.run(""" MATCH (n1:Image {name: $node_name})
                                        MATCH (n2:Image)
                                        WHERE n2.name <> $node_name
                                        WITH n1, n2, gds.alpha.similarity.euclideanDistance([n1.features], [n2.features]) AS distance
                                        ORDER BY distance ASC
                                        LIMIT 5
                                        RETURN n2.id, distance
                                    """, node_name=name)
            for record in result:
                print(record)


if __name__ == "__main__":
    print("initialize driver")
    fe: FeatureExtractor = FeatureExtractor(image_dir='../data', batch_size=1)
    fe.load_images()
    print(fe.file_list)
    img:np.ndarray = fe.batch['../data/cartman.png']
    fe.database.flush_database()
    print(fe.extract_features(img))
    fe.insertImageGraph()
    fe.spark.stop()

    fe.predict('../data/cartman.png')


