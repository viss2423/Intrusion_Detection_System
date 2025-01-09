import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Load the dataset (assuming the dataset is in the same directory as the code file)
dataset_file = "combined_data_with_10_features.csv"
dtype_mapping = {
    # Provide the data types for columns if known
    # For example:
    # 'Column1': np.int32,
    # 'Column2': np.float64,
}
combined_data = pd.read_csv(dataset_file, dtype=dtype_mapping)

X = combined_data.drop(columns=['Label']).values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(combined_data['Label'])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a deep learning model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with efficient practices
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stopping]
)

# Save the trained model
model.save('ids_deep_learning_model.h10')

# Load the saved model
loaded_model = tf.keras.models.load_model('ids_deep_learning_model.h10')

# Evaluate the loaded model
y_pred_loaded = loaded_model.predict(X_test_scaled)
y_pred_class_loaded = np.round(y_pred_loaded)
accuracy_loaded = accuracy_score(y_test, y_pred_class_loaded)
print("Accuracy (Loaded Model):", accuracy_loaded)
