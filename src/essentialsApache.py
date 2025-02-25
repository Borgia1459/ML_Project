from pyspark.sql import SparkSession
import os
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.ml.feature import FeatureHasher

class SparkDataProcessor:
    def __init__(self, app_name, master='local[*]'):
        self.app_name = app_name
        self.master = master
        self.spark = None

    def create_spark_session(self):
        self.spark = SparkSession.builder \
            .appName(self.app_name) \
            .master(self.master) \
            .getOrCreate()
        return self.spark

    def load_data(self, file_path, header=True, sep=','):
        if self.spark is None:
            raise ValueError("Spark session is not initialized. Call create_spark_session() first.")
        df = self.spark.read.csv(file_path, header=header, sep=sep)
        return df

    def process_data(self, df):
        # Show the variable types of the file
        df.printSchema()

        # Show the content of the file
        df.show()

        # Show the number of rows
        row_count = df.count()
        print(f"Number of rows: {row_count}")

        # Drop some columns that have no bearing on the model
        df = df.drop('id').drop('hour').drop('device_id').drop('device_ip')

        # Display the column names
        print("Column names:", df.columns)

        # Renaming the 'Clicked on Ad' column to 'label'
        df = df.withColumnRenamed("Clicked on Ad", "label")

        # Convert the 'label' column to numeric type
        df = df.withColumn("label", col("label").cast("double"))

        return df

    def train_test_data(self, df):
        # Split the data into training and test sets (70% and 30%)
        train, test = df.randomSplit([0.7, 0.3], 42)
        # Storing the cache in memory
        train.cache()
        train.count()
        test.cache()
        test.count()
        # Show the number of rows in each set
        print(f"Number of rows in training set: {train.count()}")
        print(f"Number of rows in test set: {test.count()}")

        return train, test

    def one_hot_encode(self, train, test):
        # One-hot encoding for categorical variables
        categorical = [c for c in train.columns if c != 'label']
        print('The following are the categorical variables:', categorical)

        # Combine train and test datasets to ensure consistent encoding
        combined_df = train.unionByName(test, allowMissingColumns=True)

        # Create indexers and encoders
        indexers = [StringIndexer(inputCol=c, outputCol=f'{c}_indexed').setHandleInvalid('keep') for c in categorical]
        encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol=f'{indexer.getOutputCol()}_encoded') for indexer in indexers]

        # Assemble features
        assembler = VectorAssembler(inputCols=[encoder.getOutputCol() for encoder in encoders], outputCol='features')

        # Create and fit the pipeline
        pipeline = Pipeline(stages=indexers + encoders + [assembler])
        one_hot_encoder = pipeline.fit(combined_df)

        # Transform both training and test data
        train_encoded = one_hot_encoder.transform(train).select('label', 'features')
        test_encoded = one_hot_encoder.transform(test).select('label', 'features')

        print(f"Train Encoded Count: {train_encoded.count()}, Test Encoded Count: {test_encoded.count()}")
        print(f"Train Encoded Schema: {train_encoded.printSchema()}")
        print(f"Test Encoded Schema: {test_encoded.printSchema()}")
        # Show the encoded DataFrames
        train_encoded.show()
        test_encoded.show()

        return train_encoded, test_encoded, categorical

    def train_test_logic_regression_mode(self, train_encoded, test_encoded):
        # Train a logistic regression model on the training data
        try:
            classifier = LogisticRegression(maxIter=20, regParam=0.001, elasticNetParam=0.001)
            lr_model = classifier.fit(train_encoded)
        
            # Make predictions on the training data
            train_predictions = lr_model.transform(train_encoded)
            train_predictions.cache()
            train_predictions.show()
            print('\n we here now! \n')
        
            # Make predictions on the test data
            test_predictions = lr_model.transform(test_encoded)
            
            # Evaluate the model on the test data
            evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",metricName='areaUnderROC')
            auc_score = evaluator.evaluate(test_predictions)
            print(f"Test AUC-ROC: {auc_score}")
        
            return train_predictions, test_predictions
        except Exception as e:
            print(f"An error occurred: {e}")

    def hashing_method(self, train, test, categorical):
        """
        Hash categorical features using FeatureHasher.

        Parameters:
        - train: DataFrame containing the training data.
        - test: DataFrame containing the test data.
        - categorical: List of categorical columns to hash.

        Returns:
        - train_hashed: DataFrame with hashed features for training data.
        - test_hashed: DataFrame with hashed features for test data.
        """
        try:
            # Validate input
            if not categorical:
                raise ValueError("No categorical columns provided for hashing.")
            
            # Initialize FeatureHasher
            hasher = FeatureHasher(numFeatures=10000, inputCols=categorical, outputCol='features')
            
            # Transform the training and test data
            train_hashed = hasher.transform(train).select('features')
            test_hashed = hasher.transform(test).select('features')
            
            # Show the hashed features
            print("Hashed Features for Training Data:")
            train_hashed.show(truncate=False)
            
            print("Hashed Features for Test Data:")
            test_hashed.show(truncate=False)
            
            #making a pipeline
            classifier= LogisticRegression(maxIter=20, regParam=0.000, elasticNetParam=0.000)
            stages= [hasher, classifier]
            pipeline = Pipeline(stages=stages)
            model = pipeline.fit(train)
            predictions = model.transform(test)
            predictions.cache()
            #AUC-ROC
            evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", metricName='areaUnderROC')
            print(f'Test AUC-ROC: {evaluator.evaluate(predictions)}')
            
            return train_hashed, test_hashed
        except Exception as e:
            print(f"An error occurred during hashing: {e}")

    def stop_spark_session(self):
        if self.spark is not None:
            self.spark.stop()

if __name__ == '__main__':
    # Check current working directory
    print("Current Working Directory:", os.getcwd())

    # Create an instance of SparkDataProcessor
    processor = SparkDataProcessor(app_name="Check Spark Version")

    # Create a Spark session
    spark = processor.create_spark_session()

    # Load the data into the Spark DataFrame using an absolute path
    file_path = 'C:/Users/phiri/Documents/ML/PMLbE/Apache/inputs/ad_records.csv'
    df = processor.load_data(file_path)

    # Process the data
    df = processor.process_data(df)

    # Split the data into training and test sets
    train, test = processor.train_test_data(df)

    # One-hot encode the categorical variables
    train_encoded, test_encoded, categorical = processor.one_hot_encode(train, test)
    processor.hashing_method(train, test, categorical)

    # Train a logistic regression model and evaluate it
    processor.train_test_logic_regression_mode(train_encoded, test_encoded)

    # Hashing method
    train_hashed, test_hashed = processor.hashing_method(train, test, categorical)

    # Stop the Spark session
    processor.stop_spark_session()
