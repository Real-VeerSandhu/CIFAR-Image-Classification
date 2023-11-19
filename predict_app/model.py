import tensorflow as tf

def fetch_model():
    return tf.keras.models.load_model('models/cifar_cnn1.h5')

def check_gpu():
    tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None) 

    