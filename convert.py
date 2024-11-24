import tensorflow as tf

# Load the .h5 model
model = tf.keras.models.load_model('model_transferlearning.h5')

# Save the model in SavedModel format
model.save('saved_model/my_model')
