import tensorflow as tf
from datetime import datetime

# reference: https://learningtensorflow.com/lesson10/
timeTakens = []

def test(device_name):
    shape = (10000, 10000)  # matrix size
    with tf.device(device_name):
        random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
        dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
        sum_operation = tf.reduce_sum(dot_operation)

    startTime = datetime.now()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation)
        #print(result)

    timeTaken = datetime.now() - startTime
    timeTakens.append(timeTaken)

test("/gpu:0")
test("/cpu:0")

print("\n")
print("GPU time taken: ")
print(timeTakens[0])
print("CPU time take: ")
print(timeTakens[1])
