import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import math

tf.reset_default_graph()
log_dir = 'logdir'

def get_data(fout) :
    [ms,mq,mt] = np.load(fout)

    # ms = [m for m in ms]
    ms = list(ms)
    mq = list(mq)
    mt = list(mt)

    i1 = math.floor(0.7 * len(ms))

    ms_train = ms[:i1]
    ms_test = ms[i1:]

    mq_train = mq[:i1]
    mq_test = mq[i1:]

    mt_train = mt[:i1]
    mt_test = mt[i1:]

    #train = tf.data.Dataset.from_tensor_slices( ( np2tf(ms_train, mq_train, mt_train) ) )
    #test = tf.data.Dataset.from_tensor_slices((np2tf(ms_test, mq_test, mt_test)))

    return (ms_train, mq_train, mt_train), (ms_test, mq_test, mt_test)

def split_list(l,frac) :
    a = []
    b = []
    n = len(l)
    for i in range( len(l) ) :
        c = i < frac*n
        if c :
            a.append(l[i])
        else :
            b.append(l[i])
    return a,b


def np2tf(ms,mq,mt):
    ms = list(map(tf.convert_to_tensor, ms))
    mt = list(map(tf.convert_to_tensor, mt))
    mq = [list(map(tf.convert_to_tensor, dm)) for dm in mq]

    return ms, mq, mt


from generate_data import test1

#fout = 'data/object_retrieve_2_0_T100.npy'
fout = test1(1000)
m_train,m_test = get_data(fout)

N_train = len(m_train[0] )
BATCH_SIZE = N_train // 1

# Convert numpy arrays to a tensorflow dataset. The arrays should be a list of Arrays.
def numpy_arrays2dataset(variables) :
    K = len(variables) # Number of variables. Each element is a list.
    var_phs = []
    for k in range(K) :
        # da = np.asarray(variables[k])
        # SZ = variables[k][0].shape
        SZ = np.asarray(variables[k][0]).shape

        print("Making dataset type of shape " + str(SZ) )
        var_phs.append(tf.placeholder(tf.float32, shape=(None,) + SZ))

    BS = tf.placeholder(tf.int64, shape=())
    dataset = tf.data.Dataset.from_tensor_slices(tuple(var_phs)).batch(tf.reshape(BS, [])).repeat()
    return dataset,BS,var_phs

dataset,BS,var_ph = numpy_arrays2dataset( m_train )

iter = dataset.make_initializable_iterator()
ms,mq,mt = iter.get_next()

Nxy,Ntypes = m_train[0][0].shape

#Ntypes = Nencoding['CLASS']
#Nxy = Nencoding['Nxy']
#Nkw = Nencoding['KEYWORD']


lookup_im = tf.matmul( ms, mq[:,0,:,:] )
comb_keyword = tf.concat([ lookup_im, mq[:,1,:,:]], 1)

return_query = tf.layers.dense(tf.transpose( comb_keyword, perm=[0,2,1]),units=Nxy, activation=tf.nn.relu)
return_query = tf.layers.dense( return_query, units=Nxy, activation=tf.nn.sigmoid)

return_img = tf.matmul(return_query,  ms)

output = tf.transpose( tf.layers.dense(return_img, units=Ntypes, activation=tf.nn.relu ), perm=[0,2,1])
#output = tf.layers.dense(return_img, units=Nencoding['CLASS']+5, activation=tf.nn.relu )
#output = tf.transpose( tf.layers.dense(output, units=Nencoding['CLASS'], activation=tf.nn.relu ), perm=[0,2,1])

mt = tf.squeeze(mt)
output = tf.squeeze(output)

loss = tf.losses.softmax_cross_entropy(onehot_labels=mt, logits=output )#, tf.transpose(mt, perm=[0, 2, 1]))

tf.summary.scalar("Training loss", loss)

#train_op = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)
#train_op = tf.train.AdagradOptimizer(learning_rate = 0.1).minimize(loss)
train_op = tf.train.GradientDescentOptimizer(learning_rate =.5).minimize(loss)

EPOCHS = 20000

correct_prediction = tf.equal(tf.argmax(mt, 1), tf.argmax(output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("Training Accuracy", accuracy)

merged = tf.summary.merge_all()

print("Now printing...")
je = 0
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(log_dir, sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + "/test", sess.graph)

    sess.run(tf.global_variables_initializer())


    print('Training...')
    n_batches = N_train // BATCH_SIZE
    for i in range(EPOCHS):

        if i % 10 == 0 :
            N_test = len(m_test[0])
            sess.run(iter.initializer, feed_dict={var_ph[0]: m_test[0], var_ph[1]: m_test[1], var_ph[2]: m_test[2], BS: N_test})
            test_loss_value, summary, test_acc = sess.run([loss, merged, accuracy])

            test_writer.add_summary(summary, i)

            sess.run(iter.initializer,
                     feed_dict={var_ph[0]: m_train[0], var_ph[1]: m_train[1], var_ph[2]: m_train[2], BS: BATCH_SIZE})


        tot_loss = 0
        for _ in range(n_batches):
            _, loss_value,summary,train_acc = sess.run([train_op, loss,merged,accuracy])

            tot_loss += loss_value

        train_writer.add_summary(summary, i)

        if i % 10 == 0:
            print("Iter: {}, Loss: {:.4f} test loss: {:.4f}, train_acc {:.4f} test_acc {:.4f}".format(i, tot_loss / n_batches,
                                                                                     test_loss_value, train_acc, test_acc))


train_writer.close()
print("END WORLD")