import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns  # Make sure to import seaborn

def plot_learning_curves(model, X_train, y_train, X_val, y_val):
    history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val), verbose=0)

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

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

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

num_features = X_train_scaled.shape[1]

# Load the pre-trained model
loaded_model = tf.keras.models.load_model('D:\Python files\Mini Project\model\deeper_hybrid_model.keras')

# Load the preprocessing objects
with open('D:\Python files\Mini Project\model\scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

with open('D:\Python files\Mini Project\model\label_encoder.pkl', 'rb') as label_encoder_file:
    loaded_label_encoder = pickle.load(label_encoder_file)

# Apply preprocessing to the test set
X_test_scaled = loaded_scaler.transform(X_test)
y_test_encoded = loaded_label_encoder.transform(y_test)

# Plot learning curves using the loaded history
plot_learning_curves(loaded_model, X_train_scaled, y_train, X_test_scaled, y_test_encoded)

# Calculate precision, recall, F1 score, and confusion matrix
y_pred = loaded_model.predict(X_test_scaled)
y_pred_binary = (y_pred > 0.5).astype(int)

precision = precision_score(y_test_encoded, y_pred_binary)
recall = recall_score(y_test_encoded, y_pred_binary)
f1 = f1_score(y_test_encoded, y_pred_binary)
confusion = confusion_matrix(y_test_encoded, y_pred_binary)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion)

# Display the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=['BENIGN', 'MALIGNANT'], yticklabels=['BENIGN', 'MALIGNANT'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
