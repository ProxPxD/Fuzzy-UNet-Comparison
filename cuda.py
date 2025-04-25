import tensorflow as tf

# Check if GPU is detected
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: " + str(is_gpu := tf.config.list_physical_devices('GPU')))

# Run a simple computation on GPU
if is_gpu:
    print("TensorFlow is using GPU!")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print(f"Matrix multiplication result:\n{c}")
else:
    print("TensorFlow is not using GPU!")