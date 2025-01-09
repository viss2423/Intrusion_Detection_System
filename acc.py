from tensorflow.keras.models import load_model, save_model

# Load the Keras model
keras_model = load_model("D:\Python files\Mini Project\model\deeper_hybrid_model.keras")

# Save the model in HDF5 format
save_model(keras_model, "hybrid_model.h5")