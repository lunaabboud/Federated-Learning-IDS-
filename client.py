import os
import flwr as fl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
import tensorflow as tf
from tensorflow.keras import regularizers

# Load training data
client_id = int(os.getenv("CLIENT_ID", 1))
#train_file = f"/app/data/train_client_{client_id}.csv" # Load data specific to each client
train_file = f"/app/data/ids{client_id}.csv" # Load data specific to each client
df = pd.read_csv(train_file)
df.dropna(inplace=True)

# --- Preprocessing ---
# 1. Drop timestamp column
df = df.drop(['Timestamp'], axis=1, errors='ignore')

# 2. Encode 'Protocol' column
le = LabelEncoder()
df['Protocol'] = le.fit_transform(df['Protocol'])

# 3. Convert 'Label' to numeric
df['Label'] = df['Label'].apply(lambda x: 1 if x == 'Benign' else 0)

# 4. Validate numeric columns
non_numeric_cols = df.select_dtypes(include='object').columns.tolist()
assert not non_numeric_cols, f"Non-numeric columns detected: {non_numeric_cols}"

# Prepare features/labels
X = df.drop("Label", axis=1).values.astype(np.float32)
y = df['Label'].values

# --- Fix: Handle extreme values and zero-variance features ---
X = np.nan_to_num(X, nan=0.0, posinf=1e4, neginf=-1e4)  # Replace NaN/Inf
X = np.clip(X, -1e4, 1e4)  # Clip to safe range

# Remove zero-variance features
selector = VarianceThreshold(threshold=0)
X = selector.fit_transform(X)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42  # Use a constant random state here
)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Verify data integrity
assert not np.isnan(X_train).any(), "Training data contains NaNs"
assert not np.isinf(X_train).any(), "Training data contains Infs"
assert not np.isnan(X_val).any(), "Validation data contains NaNs"
assert not np.isinf(X_val).any(), "Validation data contains NaNs"
assert X_train.shape[0] > 0, "Training data is empty"
assert X_val.shape[0] > 0, "Validation data is empty"

# Print label distribution and sample features
print(f"Client {client_id} label distribution:")
print(df['Label'].value_counts())
print(f"Client {client_id} sample features:")
print(X_train[:5])  # Print first 5 rows


# Define model
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01)), #Increase nodes and add regularization
        tf.keras.layers.Dropout(0.5),  # Add dropout for regularization
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01)),  # Add regularization
        tf.keras.layers.Dropout(0.5), #Add dropout
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

model = create_model((X_train.shape[1],))

# Flower client
class IDSClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        history = model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0) #Increased to 10 epochs
        print(f"Client {client_id} training loss: {history.history['loss'][-1]:.4f}", flush=True)
        return model.get_weights(), len(X_train), {"accuracy": history.history['accuracy'][-1], "client_id": client_id}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"Client {client_id} validation accuracy: {accuracy*100:.2f}%", flush=True)
        return loss, len(X_val), {"accuracy": accuracy, "client_id": client_id}

if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="server:8080",
        client=IDSClient(),
    )

