
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model(learning_rate: float = 0.001):
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):
    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)
    x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]

    return x_train, y_train, x_test, y_test



'''
def load_model(learning_rate: float = 0.001):
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(512, activation='relu', input_shape=(26,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer without activation
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


fds = None  # Cache FederatedDataset


def preprocess_data(data: pd.DataFrame):
    """Preprocess the dataset."""
    label_encoders = {}
    for column in ["gender", "tartar", "oral"]:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Handle outliers by clipping extreme values
    clip_columns = ["Gtp", "AST", "ALT"]
    for column in clip_columns:
        upper_limit = np.percentile(data[column], 99)
        data[column] = np.clip(data[column], None, upper_limit)

    # Split into features (X) and labels (y)
    X = data.drop(columns=["smoking"])  # "smoking" is the target column
    y = data["smoking"]

    # Normalize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values


def load_data(partition_id: int, num_partitions: int):
    """Load and partition the local dataset."""
    # Load the dataset
    data = pd.read_csv("./smoking.csv")

    # Preprocess the data
    X, y = preprocess_data(data)

    # Split data into partitions
    data_size = len(y)
    partition_size = data_size // num_partitions

    # Get the partition data for this client
    start_idx = partition_id * partition_size
    end_idx = (partition_id + 1) * partition_size
    x_partition = X[start_idx:end_idx]
    y_partition = y[start_idx:end_idx]

    # Further split the partition into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_partition, y_partition, test_size=0.2, random_state=42
    )

    return x_train, y_train, x_test, y_test
'''