import cv2
import numpy as np
import os
from pyspark.sql import SparkSession
from PIL import Image, ImageFilter
import mahotas
from .database import *
from collections import defaultdict
import pickle

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
        self.data: list = []
        self.labels:list = []   

        if 'cifar' in image_dir:
            batch_files:list = ['data_batch_1', 'data_batch_2', 'data_batch_3']
            for batch_file in batch_files:
                batch_file_path = os.path.join(image_dir, batch_file)


                with open(batch_file_path, 'rb') as fo:
                    # Load the batch data using pickle
                    batch_data = pickle.load(fo, encoding='bytes')


                    # Extract image data and labels from batch data
                    self.data.extend(batch_data[b'data'])
                    self.labels.extend(batch_data[b'labels'])
            with open(f'{image_dir}/batches.meta', 'rb') as f:
                meta = pickle.load(f, encoding='bytes')


            class_labels = meta[b'label_names']
            lbls: list = [label.decode('utf-8') for label in class_labels]


            self.labels =  [lbls[i] for i in self.labels][:len(self.data)]

        else:
            # Iterate over all files in the directory
            for filename in os.listdir(image_dir):

                filepath: str = os.path.join(image_dir, filename)
            
                if os.path.isfile(filepath):
                    # Add the file path to the list
                    self.file_list.append(filepath)

        

     # define a function to load images in batches and convert them to numpy arrays using spark 
    def load_images(self) -> None:
        images_rdd = self.spark.sparkContext.parallelize(self.file_list)

        def load_image_np(img_path) -> tuple:
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

    def load_cifar(self) -> None:
       images_rdd = self.spark.sparkContext.parallelize(self.data)

       def load_image_np(batch) -> tuple:
           # convert the image to a 32x32 numpy array and apply filtering
           img: np.ndarray = cv2.resize(batch, (32, 32))
           # img: np.ndarray = cv2.cvtColor(batch, cv2.COLOR_BGR2GRAY)
           img = cv2.GaussianBlur(img, (5,5), 0)
           img = cv2.medianBlur(img, 5)


           return (batch, img)

       images_rdd = images_rdd.map(load_image_np).filter(lambda x: x[1] is not None)


       images = images_rdd.collect()
       self.batch: dict = dict(zip(self.labels, images))
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
        return [np.mean(img), np.std(img), np.median(img), np.min(img), np.max(img), np.corrcoef(img)[0][0], np.cov(img)[0][0]]

    def insertImageGraph(self, image_types='local') -> None:
        if self.batch is None:
            if image_types == 'cifar':
                self.load_cifar()
            else:
                self.load_images()

        for label, img in self.batch.items():
            if 'cat' in label.lower():
                label:str = 'cat'
            if 'dog' in label.lower():
                label:str = 'dog'
            print(label)
            try:
                feat_array: list = self.extract_features(img)
            except:
                feat_array: list = self.extract_features(np.asarray(img[1]))
            print(feat_array)


            with self.database.driver.session() as session:
                # Create the image node
                session.write_transaction(
                    lambda tx: tx.run("CREATE (:Image {name: $name, mean: $mean, std: $std, median: $median, min: $min, max: $max, corrcoef: $corrcoef, covariance: $cov})", name=label, mean=float(feat_array[0]), std=float(feat_array[1]), median=float(feat_array[2]), min=float(feat_array[3]), max=float(feat_array[4]), corrcoef=float(feat_array[5]), cov=float(feat_array[6]))
                )

    def initCentroids(self, k=2) -> None:
        query:str = f"""MATCH (n)
                       WITH n, rand() as r
                       ORDER BY r
                       LIMIT {k}
                       CREATE (:Centroid {{mean: n.mean, std: n.std, median: n.median, min: n.min, max: n.max, corrcoef: n.corrcoef, covariance: n.covariance}})
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
            
    def removeConnections(self) -> None:
        query: str = """
            MATCH ()-[r]-()
            DELETE r
        """
        with self.database.driver.session() as session:
            session.run(query)
    
    def getAllNodes(self):
        # return all centroids and images
        query_get_centroids: str = """
                        MATCH (c:Centroid)
                        RETURN c
                    """
        query_get_images: str = """
                        MATCH (i:Image)
                        RETURN i
                    """
        
        with self.database.driver.session() as session:
            centroids = session.run(query_get_centroids)
            centroid_id_to_properties = {}
            for record in centroids:
                centroid_id_to_properties[record["c"]._id] = record["c"]._properties
            
            images = session.run(query_get_images)
            image_id_to_properties = {}
            for record in images:
                image_id_to_properties[record["i"]._id] = record["i"]._properties
            
            return centroid_id_to_properties, image_id_to_properties
        
    def connectToCentroid(self, centroid_id_to_properties, image_id_to_properties):
        def cosine_similarity(x, y):
            return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        
        # connect images to centroids based on cosine similarity
        for image_id, image_properties in image_id_to_properties.items():
            # delete the image name from the properties
            del image_properties["name"]
            
            closest_centroid_id = None
            closest_centroid_similarity = np.inf
            for centroid_id, centroid_properties in centroid_id_to_properties.items():
                image_features = list(image_properties.values())
                centroid_features = list(centroid_properties.values())
                
                # calculate cosine similarity
                similarity = cosine_similarity(image_features, centroid_features)
                
                # update the closest centroid
                if similarity < closest_centroid_similarity:
                    closest_centroid_id = centroid_id
                    closest_centroid_similarity = similarity

            # connect the image to the closest centroid and store the cosine similarity
            print(f"Connecting image {image_id} to centroid {closest_centroid_id} with similarity {closest_centroid_similarity}")
            
            query: str = """
                            MATCH (i:Image) WHERE ID(i)=$image_id
                            MATCH (c:Centroid) WHERE ID(c)=$centroid_id
                            CREATE (i)-[r:CLOSEST_TO]->(c)
                            SET r.cosine_similarity = $similarity
                        """
            with self.database.driver.session() as session:
                session.run(query, image_id=image_id, centroid_id=closest_centroid_id, similarity=closest_centroid_similarity)

    def recalcCentroid(self) -> None:
        with self.database.driver.session() as session:
            # get all the centroids
            query_get_centroids: str = """
                                    MATCH (c:Centroid)
                                    RETURN c
                                """
            
            # get a list of centroid ids
            centroids = session.run(query_get_centroids)
            centroid_id_to_properties = {}
            for record in centroids:
                centroid_id_to_properties[record["c"]._id] = record["c"]._properties

            # get all images connected to a centroid
            query_get_nodes: str = """
                                MATCH (i:Image)-[:CLOSEST_TO]->(c:Centroid) WHERE ID(c)=$centroid_id
                                RETURN i
                                """
            
            # recalculate centroid averages
            for centroid_id, centroid_properties in centroid_id_to_properties.items():
                # get all the nodes connected to the centroid
                nodes = session.run(query_get_nodes, centroid_id=centroid_id)
                node_feature_sums = defaultdict(int)
                
                centroid_features = list(centroid_properties.keys())
                num_nodes = 0
                
                # loop through nodes to get sum of all the nodes features
                for node in nodes:
                    num_nodes += 1
                    for feature in centroid_features:
                        node_feature_sums[feature] += node["i"]._properties[feature]
                
                # calculate the mean for each feature
                corrcoef = node_feature_sums["corrcoef"] / num_nodes
                covariance = node_feature_sums["covariance"] / num_nodes
                max = node_feature_sums["max"] / num_nodes
                mean = node_feature_sums["mean"] / num_nodes
                median = node_feature_sums["median"] / num_nodes
                min = node_feature_sums["min"] / num_nodes
                std = node_feature_sums["std"] / num_nodes
                
                # print old and new centroid features
                print(f"Old centroid features: corrcoef={centroid_properties['corrcoef']}, covariance={centroid_properties['covariance']}, max={centroid_properties['max']}, mean={centroid_properties['mean']}, median={centroid_properties['median']}, min={centroid_properties['min']}, std={centroid_properties['std']}")
                print(f"New centroid features: corrcoef={corrcoef}, covariance={covariance}, max={max}, mean={mean}, median={median}, min={min}, std={std}")
                
                # update the centroid
                query_update_centroid: str = """
                                            MATCH (c:Centroid) WHERE ID(c)=$centroid_id
                                            SET c.corrcoef = $corrcoef, c.covariance = $covariance, c.max = $max, c.mean = $mean, c.median = $median, c.min = $min, c.std = $std
                                            """
                session.run(query_update_centroid, centroid_id=centroid_id, corrcoef=corrcoef, covariance=covariance, max=max, mean=mean, median=median, min=min, std=std)
                
    def count_connections(self):
        query: str = """
                        MATCH (i:Image)-[r:CLOSEST_TO]->(c:Centroid)
                        RETURN c, count(r)
                    """
        with self.database.driver.session() as session:
            result = session.run(query)
            id_to_count = {}
            for record in result:
                id_to_count[record["c"]._id] = record["count(r)"]
            return id_to_count
        
    def scoreClusters(self):
        # get list of names of all nodes connected to each centroid
        query: str = """
                        MATCH (i:Image)-[r:CLOSEST_TO]->(c:Centroid)
                        RETURN ID(c), i.name
                    """
        with self.database.driver.session() as session:
            result = session.run(query)
            id_to_names = defaultdict(list)
            for record in result:
                id_to_names[record["ID(c)"]].append(record["i.name"])
                
            # calculate the silhoutte score for each centroid
            id_to_score = {}
            for centroid_id, names in id_to_names.items():
                # for each cluster calculate the # of type a - type b / total number of nodes
                print(names)
                dog_count = sum([1 for name in names if "dog" in name])
                cat_count = sum([1 for name in names if "cat" in name])
                diff = abs(dog_count - cat_count)
                total = dog_count + cat_count
                score = diff / total
                id_to_score[centroid_id] = score
            
            return id_to_score
    
    def train(self) -> None:
        # remove all current connections
        self.removeConnections()
        centroid_id_to_properties, image_id_to_properties = self.getAllNodes()
        self.connectToCentroid(centroid_id_to_properties, image_id_to_properties)
        id_to_count = self.count_connections()
        
        # continue clustering until the clusters are stable
        iter:int = 0
        while True:
            # recalculate centroids
            self.recalcCentroid()
            
            # remove all current connections
            self.removeConnections()
            
            # get all the nodes again
            centroid_id_to_properties, image_id_to_properties = self.getAllNodes()
            
            # connect the images to the centroids
            self.connectToCentroid(centroid_id_to_properties, image_id_to_properties)
            
            # get the new counts
            new_id_to_count = self.count_connections()
            
            cluster_score = self.scoreClusters()
            print(f"Cluster score: {cluster_score}")
            
            # check if the counts are the same
            if id_to_count == new_id_to_count:
                break
            else:
                id_to_count = new_id_to_count
            iter+=1
            if iter>100:
                break
        
        self.recalcCentroid()
        
        
