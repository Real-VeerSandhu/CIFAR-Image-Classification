import tensorflow as tf

def fetch_model():
    return tf.keras.models.load_model('models/cifar_cnn1.h5')