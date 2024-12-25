import subprocess
import flwr as fl
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


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
                tf.keras.layers.Dense(1)  # Output layer without activation
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
                epochs=20,
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


def return_pruned_dataset(data):
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Compute the correlation matrix
    correlation_matrix = data.corr()
    # Create a heatmap using seaborn
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    # Add title
    plt.title("Correlation Matrix Heatmap", fontsize=16)
    # Show the plot
    plt.tight_layout()
    plt.show()
    selected_features = correlation_matrix['smoking'][abs(correlation_matrix['smoking']) >= 0.3].index.tolist()
    selected_features.remove('smoking')  # Remove the target variable itself
    print(selected_features)
    return data.drop(columns=selected_features)


# Start Flower client
def start_flower_client():
    flower_client = subprocess.Popen(
            ["flower-supernode", "--insecure", "--superlink=localhost:8080"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
    )
    return flower_client

# Start Flower server
def start_flower_server():
    flower_server = subprocess.Popen(
            ["flower-superlink", "--insecure", "--grpc-server-address=0.0.0.0:8080"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
    )
    return flower_server


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
        
    data = return_pruned_dataset(data.drop(columns=['ID']))    

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
    clients = []
    for X_train, y_train, X_test, y_test in client_data:
        clients.append(FederatedClient(X_train, y_train, X_test, y_test))
    flower_server = start_flower_server()
    print("Flower server started using SuperLink CLI.")
    flower_client = start_flower_client()
    print("Flower client started using SuperNode CLI.")

    # Global evaluation
    all_predictions = []
    all_actuals = []
    for client in clients:
        actual = client.y_test
        predictions = client.model.predict(client.X_test) > 0.5
        all_predictions.extend(predictions)
        all_actuals.extend(actual)
    # Compute global accuracy
    from sklearn.metrics import accuracy_score, roc_auc_score
    accuracy = accuracy_score(all_actuals, all_predictions)
    auc = roc_auc_score(all_actuals, all_predictions)
    print(f"Global Accuracy: {accuracy * 100:.2f}%")
    print(f"Global AUC: {auc:.2f}")
    # Stop server and client
    flower_client.terminate()
    flower_client.wait()
    flower_server.terminate()
    flower_server.wait()
    print("Flower server and client stopped.")



main()