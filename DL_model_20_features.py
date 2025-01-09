import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load the dataset (assuming the dataset is in the same directory as the code file)
dataset_file = "D:\Python files\Mini Project\combined_data.csv"
dtype_mapping = {
    # Provide the data types for columns if known
    # For example:
    # 'Column1': np.int32,
    # 'Column2': np.float64,
}
combined_data = pd.read_csv(dataset_file, dtype=dtype_mapping)

# Map all non-'Benign' labels to 'Malignant'
combined_data['Label'] = np.where(combined_data['Label'] != 'BENIGN', 'MALIGNANT', 'BENIGN')

X = combined_data.drop(columns=['Label']).values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(combined_data['Label'])

# Split dataset into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

num_features = X_train_scaled.shape[1]

# Create a deeper hybrid CNN-RNN model for tabular data
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(num_features,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Reshape((16, 8)),  # Reshape for input to RNN
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.LSTM(64, activation='relu', return_sequences=True),
    tf.keras.layers.LSTM(64, activation='relu', return_sequences=True),
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Implement early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping and learning curve collection
history = model.fit(
    X_train_scaled, y_train, 
    epochs=50, 
    batch_size=64, 
    validation_split=0.1, 
    callbacks=[early_stopping],
    verbose=1  # Set verbose to 1 to see the training progress
)

# Collect the training and validation loss and accuracy
training_loss = history.history['loss']
training_accuracy = history.history['accuracy']
validation_loss = history.history['val_loss']
validation_accuracy = history.history['val_accuracy']

# Evaluate the model
accuracy = model.evaluate(X_test_scaled, y_test)[1]
print("Test Accuracy:", accuracy)

# Calculate precision, recall, F1 score, and confusion matrix
y_pred = model.predict(X_test_scaled)
y_pred_binary = (y_pred > 0.5).astype(int)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)
confusion = confusion_matrix(y_test, y_pred_binary)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion)

# Determine model behavior based on learning curves
if training_loss[-1] > validation_loss[-1]:
    print("Possibly underfitting. Consider increasing model capacity.")
elif training_loss[-1] < validation_loss[-1]:
    print("Possibly overfitting. Consider reducing model capacity or using regularization.")
else:
    print("Good fit. Continue training or consider stopping.")

# Plot learning curves
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_validation_loss.png')  # Save the figure

plt.subplot(1, 2, 2)
plt.plot(training_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_validation_accuracy.png')  # Save the figure

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.savefig('roc_curve.png')  # Save the figure

plt.show()
