from ..graphnn.extract_features import FeatureExtractor

if __name__ == "__main__":
    print("initialize driver")

    fe: FeatureExtractor = FeatureExtractor(image_dir='data', batch_size=1)
    fe.spark.stop()
    print('initialized driver')