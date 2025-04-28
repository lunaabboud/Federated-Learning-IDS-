import flwr as fl
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes

# --- Load and Preprocess Test Data ---
#test_df = pd.read_csv("/app/data/test.csv")
test_df = pd.read_csv("/app/data/IDS.csv")
test_df.dropna(inplace=True)

# Drop timestamp column
test_df = test_df.drop(['Timestamp'], axis=1, errors='ignore')

# Encode 'Protocol' and convert 'Label'
le = LabelEncoder()
test_df['Protocol'] = le.fit_transform(test_df['Protocol'])
test_df['Label'] = test_df['Label'].apply(lambda x: 1 if x == 'Benign' else 0)

# Validate numeric columns
non_numeric_cols = test_df.select_dtypes(include='object').columns.tolist()
assert not non_numeric_cols, f"Non-numeric columns detected: {non_numeric_cols}"

# Prepare test data
X_test = test_df.drop("Label", axis=1).values.astype(np.float32)
y_test = test_df['Label'].values

# --- Fix: Clip, clean, and normalize data ---
X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e4, neginf=-1e4)
X_test = np.clip(X_test, -1e4, 1e4)
selector = VarianceThreshold(threshold=0)
X_test = selector.fit_transform(X_test)
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

#Print Test Data Label Distribution
print("Test data label distribution:")
print(test_df['Label'].value_counts())
print("Test data sample features:")
print(X_test[:5])

# --- Define Model AFTER X_test is initialized ---
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=input_shape),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# --- Initialize parameters using the model ---
initial_model = create_model((X_test.shape[1],))
initial_parameters = fl.common.ndarrays_to_parameters(initial_model.get_weights())

# Custom strategy
class TestEvalStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameters = initial_parameters  # Initialize with valid parameters

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        if self.parameters is not None:
            model = create_model((X_test.shape[1],))
            weights = fl.common.parameters_to_ndarrays(self.parameters)
            model.set_weights(weights)
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"\nGlobal model test accuracy before Aggregation round {server_round}: {accuracy*100:.2f}%")
        return aggregated_loss, aggregated_metrics

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average and print client accuracies."""
        # Aggregate model parameters
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # Print client accuracies
        print(f"\nRound {server_round} client accuracies:")
        for client_proxy, fit_res in results:
            client_id = fit_res.metrics.get("client_id")  #Extract the client ID from metrics
            accuracy = fit_res.metrics.get("accuracy")
            if accuracy is not None:
                print(f"  Client {client_id}: {accuracy:.4f}", flush=True)
            else:
                print(f"  Client {client_id}: Accuracy not provided")

        # Evaluate global model after aggregation
        if aggregated_parameters is not None:
            model = create_model((X_test.shape[1],))
            weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
            model.set_weights(weights)
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"  Global model test accuracy after Aggregation round {server_round}: {accuracy*100:.2f}%")

        return aggregated_parameters, aggregated_metrics


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Dict[str, fl.common.Scalar]]]) -> Dict[str, fl.common.Scalar]:
    """Aggregate metrics using weighted average."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    aggregated_accuracy = sum(accuracies) / sum(examples)
    return {"accuracy": aggregated_accuracy}

# Start server
if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=TestEvalStrategy(
            min_available_clients=2,
            min_fit_clients=2,
            evaluate_metrics_aggregation_fn=weighted_average,
        ),
        config=fl.server.ServerConfig(num_rounds=10), # Increased to 10 rounds
    )

