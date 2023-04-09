from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import cv2
import numpy as np
from ..database.imageDB import ImageDB
import pyspark
from tqdm import tqdm

class FeatureExtractor():
    def __init__(self, image_dir, batch_size=5) -> None:
        # create a SparkSession for our feature extractor
        self.spark: SparkSession = SparkSession.builder.appName("FeatureExtractor").getOrCreate()

        # define the image directory and batch size
        self.image_dir:str = image_dir
        self.batch_size: int = batch_size

        self.batch_index:int = 0
        self.batch = None

        # connect to the Neo4j database
        self.database: ImageDB = ImageDB("neo4j://localhost:7687", "neo4j", "password")
        self.database.connect()

    # define a function to load images in batches and convert them to numpy arrays
    def load_images(self) -> None:
        images:dict = {}
        for i in tqdm(range(self.batch_size)):
            # load the image
            img_path:str = self.image_dir + "/" + batch[i]  #TODO: add directory image indexing
            img = cv2.imread(img_path)

            # convert the image to a 1028x1028 numpy array
            img: np.ndarray = cv2.resize(img, (1028, 1028))
            img: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images[img_path] = img
            self.batch_index += 1
        self.batch: list = images
    
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
    def extract_features(img: np.ndarray) -> np.ndarray:
        # Extract statistical features: TODO: add more
        return np.ndarray([np.mean(img), np.std(img), np.median(img), np.min(img), np.max(img)])

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

        feature_names: list[str] = ["Mean", "Standard Deviation", "Median", "Minimum Value", "Maximum Value"]
        for label, img in self.batch.items():
            feat_array: np.ndarray = self.extract_features(img)
            corr_matrix: np.ndarray = np.corrcoef(feat_array)

            with self.database.driver.session() as session:
                # Create the image node
                session.write_transaction(
                    lambda tx: tx.run("CREATE (:Image {label: $label})", label=label)
                )
                
                # Create nodes for each feature and add relationships between them based on the correlation matrix
                for i in range(5):
                    # Create the node for the current feature
                    feature_node:dict = {"name": feature_names[i], "value": feat_array[i]}
                    session.write_transaction(
                        lambda tx: tx.run(
                            "CREATE (n:Feature {name: $name, value: $value})-[:FROM_IMAGE]->(i:Image {label: $label})",
                            name=feature_node['name'], value=feature_node['value'], label=label)
                    )
                    
                    # Create edges between the current feature and all other features based on the correlation matrix
                    for j in range(5):
                        # Skip self-edges
                        if i == j:
                            continue
                        
                        # Get the correlation value between the current feature and the other feature
                        corr_value:float = corr_matrix[i][j]
                        
                        # Create the edge only if the correlation value is non-zero
                        if corr_value != 0:
                            # Create the edge from the current feature to the other feature
                            other_feature_node: dict[str, str] = {"name": feature_names[j]}
                            session.write_transaction(
                                lambda tx: tx.run(
                                    "MATCH (n1:Feature {name: $name1, value: $value1}), (n2:Feature {name: $name2})"
                                    "CREATE (n1)-[:CORRELATES {value: $value}]->(n2)",
                                    name1=feature_node['name'], value1=feature_node['value'],
                                    name2=other_feature_node['name'], value2=feat_array[j],
                                    value=corr_value)
                            )


