import sys
sys.path.append('/Users/brianreicher/Documents/GitHub/graph-nn/kmeans_clustering')
from kmeans_clustering import image_classification

if __name__ == "__main__":
    print("initialize driver")
    fe: image_classification.FeatureExtractor = image_classification.FeatureExtractor(image_dir='./data/cifar', batch_size=1)
    fe.load_cifar()
    fe.database.flush_database()

    fe.insertImageGraph()
    fe.initCentroids()
    fe.train()
    fe.spark.stop()
