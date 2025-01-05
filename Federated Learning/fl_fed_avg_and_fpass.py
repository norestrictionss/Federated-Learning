import subprocess
import flwr as fl
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score


# Define the Federated Client class
class FederatedClient(fl.client.NumPyClient):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(self.X_train.shape[1],)),
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
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name="auc")]
        )
        return model

    def get_parameters(self):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=5,
            batch_size=32,
            verbose=1,
            validation_split=0.1,
            callbacks=[early_stopping],
        )
        return self.get_parameters(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, auc = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return loss, len(self.X_test), {"accuracy": accuracy, "auc": auc}


# Federated Server with FedAvg
class FederatedServer:
    def __init__(self, clients):
        self.clients = clients
        self.global_weights = None  # Initialize global model weights

    def federated_averaging(self, client_weights, client_sizes):
        # Perform weighted averaging of client weights
        new_weights = []
        total_data = sum(client_sizes)

        for layer_weights in zip(*client_weights):
            avg_layer_weights = np.sum(
                [client_size * layer_weight for layer_weight, client_size in zip(layer_weights, client_sizes)],
                axis=0
            ) / total_data
            new_weights.append(avg_layer_weights)

        return new_weights

    def train(self, epochs):
        
        if self.global_weights is None:
            self.global_weights = self.clients[0].model.get_weights()
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            client_weights = []
            client_sizes = []

            for client in self.clients:
                # Train each client
                if self.global_weights is not None:
                    client.set_parameters(self.global_weights)
                weights, num_samples, _ = client.fit(self.global_weights, {})
                client_weights.append(weights)
                client_sizes.append(num_samples)

            # Perform federated averaging to update global weights
            self.global_weights = self.federated_averaging(client_weights, client_sizes)
            print("Global weights updated.")

            # Evaluate the global model
            global_loss, global_accuracy, global_auc = self.evaluate()
            print(f"Global Loss: {global_loss:.4f}, Accuracy: {global_accuracy * 100:.2f}%, AUC: {global_auc:.4f}")

    def evaluate(self):
        all_predictions = []
        all_actuals = []

        for client in self.clients:
            # Set global weights to all clients
            client.set_parameters(self.global_weights)
            loss, _, metrics = client.evaluate(self.global_weights, {})
            predictions = client.model.predict(client.X_test) > 0.5
            all_predictions.extend(predictions.flatten())
            all_actuals.extend(client.y_test)

        # Compute global metrics
        global_accuracy = accuracy_score(all_actuals, all_predictions)
        global_auc = roc_auc_score(all_actuals, all_predictions)
        return loss, global_accuracy, global_auc



# Main function
def main():
    # Load the dataset
    data = pd.read_csv("./smoking.csv")

    # Preprocess the data
    label_encoders = {}
    for column in ['gender', 'tartar', 'oral']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Handle outliers by clipping extreme values
    clip_columns = ['Gtp', 'AST', 'ALT']
    for column in clip_columns:
        upper_limit = np.percentile(data[column], 99)
        data[column] = np.clip(data[column], None, upper_limit)

    # data = return_pruned_dataset(data.drop(columns=['ID']))

    X = data.drop(columns=['smoking'])
    y = data['smoking']

    # Normalize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Simulate federated learning by splitting data among clients
    n_clients = 5
    sss = StratifiedShuffleSplit(n_splits=n_clients, test_size=0.2, random_state=42)
    client_data = []
    for train_index, test_index in sss.split(X_scaled, y):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        client_data.append((X_train, y_train, X_test, y_test))

    # Initialize clients
    clients = [FederatedClient(X_train, y_train, X_test, y_test) for X_train, y_train, X_test, y_test in client_data]

    # Initialize Federated Server
    server = FederatedServer(clients)

    # Train and evaluate
    server.train(epochs=5)  # Define the number of epochs
    print("Federated training completed.")


# Run the main function
if __name__ == "__main__":
    main()