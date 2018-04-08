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

WIN = False
if sys.platform == 'win32':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    WIN = True

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

from utils import augment
import matplotlib.pyplot as mpl



def cnn_model_fn(features, labels, mode,params):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    # iw = params['iw'] # image width (default, i.e. 28)
    Kmnist = params['Kmnist'] # Number of classes
    tw = params['tw'] # Total image width
    if isinstance(features, tf.data.Dataset):
        dataset = features
    else :
        dataset = None

    images = features['x']

    if dataset is not None :
        dw = 6
        dataset = dataset.batch(dw*dw+1)
        iterator = dataset.make_initializable_iterator()

        sess = tf.Session()
        print("Global init")
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        sess.run(iterator.initializer)
        images, labels = iterator.get_next()

    if images.shape[1].value == 28*28:
        iw = 28
        pad_size = (tw, tw)
    else:
        iw = tw
        pad_size = None

    images = tf.reshape(images,[-1,iw,iw,1])
    ## make compatible with MNIST in 28 x 28 format.



    if mode == tf.estimator.ModeKeys.PREDICT:  # Turn off input perturbation.
        class_translation_minmax = None
    else :
        class_translation_minmax = []
        for k in range(Kmnist):
            dt = [0, (tw - iw)//2 + iw //3 ]
            if k == Kmnist-1 and Kmnist == 11:
                dt = [(tw + iw) // 2 - iw // 2, (tw  + iw) // 2]

            class_translation_minmax.append( dt )
        class_translation_minmax = np.asarray(class_translation_minmax, dtype=np.float32)

    [images, labels] = augment(images, labels, resize=None, pad_to_size=pad_size, class_translation_minmax=class_translation_minmax)

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
    #dense2 = tf.layers.dense(inputs=dense1, units=1024, activation=tf.nn.relu, name="densel2")

    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=Kmnist)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
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

def get_mnist_estimator(Kmnist,tw) :
    iw = 28
    #Kmnist = 11  # max number of classes

    mod_base = "tfmodels/mnist_convnet_K%i_tw%i" % (Kmnist, tw)
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=mod_base, params={'Kmnist': Kmnist, 'iw': iw, 'tw': tw})
    print("Done!")

    return mnist_classifier

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

    batch_size = 100
    train_steps = 100000
    if WIN :
        #batch_size = 100
        train_steps = 3
        train_data = train_data[:1000,:]
        eval_data = train_data[:1000, :]
        train_labels = train_labels[:1000]
        eval_labels = eval_labels[:1000]

    K = len(np.unique(eval_labels))
    if Kmnist == 11 :
        eval_data,eval_labels = add_junk(train_data, train_labels,K)
        train_data, train_labels = add_junk(eval_data, eval_labels,K)

    model_dir = "tfmodels"
    if WIN :
        model_dir = 'tmp'
    mod_base = "%s/mnist_convnet_K%i_tw%i"%(model_dir,Kmnist,tw)

    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=mod_base,params={'Kmnist' : Kmnist, 'iw' : iw, 'tw' : tw})
    print("Done!")

    tensors_to_log = {}
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
        steps=train_steps,
        hooks=[logging_hook])
    print("Done!")

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        num_epochs=1,
        shuffle=False)
    predict_results = mnist_classifier.predict(input_fn=predict_input_fn)

    print(eval_results)

class Mnist_Wrapper() :
    def __init__(self,Kmnist=11,tw=40):
        self.mnist_classifier =get_mnist_estimator(Kmnist,tw)
        #tensors_to_log = {"probabilities": "softmax_tensor"}
        #logging_hook = tf.train.LoggingTensorHook(
        #    tensors=tensors_to_log, every_n_iter=50)

        #self.hooks = logging_hook

    def predict(self,eval_data):
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            num_epochs=1,
            shuffle=False)
        predict_results = self.mnist_classifier.predict(input_fn=predict_input_fn)

        cls = []
        probs = []
        for x in predict_results:
            cls.append(x['classes'])
            probs.append(x['probabilities'])
            print(x)
        cls = np.asarray(cls)
        probs = np.asarray(probs)
        return cls, probs

if __name__ == "__main__":
    tf.app.run()
