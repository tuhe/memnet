#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
if sys.platform == 'win32':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from generate_data import safepaste
# import matplotlib as plt
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)

from utils import augment
import matplotlib.pyplot as mpl



def cnn_model_fn(features, labels, mode,params):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    iw = params['iw'] # image width (default, i.e. 28)
    Kmnist = params['Kmnist'] # Number of classes
    tw = params['tw'] # Total image width
    if isinstance(features, tf.data.Dataset):
        dataset = features
    else :
        dataset = None

    images = features['x']

    if dataset is not None :
        print("Hacky sacky...")
        dw = 6
        dataset = dataset.batch(dw*dw+1)
        iterator = dataset.make_initializable_iterator()

        sess = tf.Session()
        print("Global init")
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        sess.run(iterator.initializer)
        images, labels = iterator.get_next()

    images = tf.reshape(images,[-1,iw,iw,1])

    class_translation_minmax = []
    for k in range(Kmnist):
        dt = [0, (tw - iw)//2 + iw //3 ]
        if k == Kmnist-1 and Kmnist == 11:
            dt = [(tw + iw) // 2 - iw // 2, (tw  + iw) // 2]

        class_translation_minmax.append( dt )
    class_translation_minmax = np.asarray(class_translation_minmax, dtype=np.float32)

    [images, labels,RT, ar] = augment(images, labels, resize=None, pad_to_size=(tw,tw), class_translation_minmax=class_translation_minmax)

    if dataset is not None:
        im,la,RTr, arr = sess.run([images,labels, RT, ar])
        for i in range(dw*dw):
            mpl.subplot(dw, dw, i+1)
            a = im[i].squeeze()
            mpl.imshow(a)
            mpl.title("Class: " + str( la[i]) )

        mpl.show()
        print("Show figures done.")

    input_layer = images
    #input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # images = tf.placeholder(tf.uint8, shape=(None, None, None)) # add ,3) as last dim for RGB.
    # labels = tf.placeholder(tf.uint64, shape=(None))

    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32] # 40 --> 20?
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2,name="pool1")

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,name="conv2")

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64] # 20 --> 10
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2,name="pool2")

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]

    nf = tw // 4

    pool2_flat = tf.reshape(pool2, [-1, nf * nf * 64],name="pool2_flat")

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense1 = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu,name="densel1")
    dense2 = tf.layers.dense(inputs=dense1, units=1024, activation=tf.nn.relu, name="densel2")

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=Kmnist)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        ## Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        ## `logging_hook`.
        # "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    iw = 28
    tw = 40
    Kmnist = 11  # max number of classes

    tf.reset_default_graph()
    print("Loading MNIST dataset...")
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)



    def add_junk(X,y,K) :
        m = eval_data.shape[0] // K
        I = np.random.randint(0, eval_data.shape[0], m)
        X2 = np.concatenate((X, X[I, :]), 0)
        y2 = np.concatenate((y, np.zeros(m)+K), 0).astype(np.int32)
        return X2, y2

    K = len(np.unique(eval_labels))
    if Kmnist == 11 :
        eval_data,eval_labels = add_junk(train_data, train_labels,K)
        train_data, train_labels = add_junk(eval_data, eval_labels,K)

    if False :
        dataset = tf.data.Dataset.from_tensor_slices((eval_data, eval_labels))

        print("Entering cnn_model_fn")
        cnn_model_fn(None,None, tf.estimator.ModeKeys.TRAIN,dataset)


        print("Done")



    mod_base = "tmp/mnist_convnet_K%i_tw%i"%(Kmnist,tw)
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=mod_base,params={'Kmnist' : Kmnist, 'iw' : iw, 'tw' : tw})
    print("Done!")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    print("Entering train mode...")
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    print("Running train..")
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=60000,
        hooks=[logging_hook])
    print("Done!")
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


if __name__ == "__main__":
    tf.app.run()
