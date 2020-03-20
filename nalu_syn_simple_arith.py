# Imports
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf
print(tf.__version__)
from sklearn import datasets
from sklearn.model_selection import KFold
import random
import os
import time
import argparse
import nalu_architectures
import ast
from nalu_architectures import *
from scipy.stats import truncnorm

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", dest="output", default="naly_syn_simple_arith")
parser.add_argument("-d", "--dist", dest="dist", default="normal", help="Prob.Dist")
parser.add_argument("-p", "--params",dest="params" , default="(-3,3)", type=ast.literal_eval)
parser.add_argument("-e", "--ext",dest="ext" , default="(10,15)", type=ast.literal_eval)

parser.add_argument("-n", "--nalu", dest="nalu", default="nalui1")
parser.add_argument("-se", "--seed", dest="seed", default=42, type=int)
parser.add_argument("-op", "--operation", dest="op", default="MUL")



args = parser.parse_args()

if args.nalu == "nalui2":
    nalu = nalui2_layer
if args.nalu == "nalui1":
    nalu = nalui1_layer
elif args.nalu == "nalum":
    nalu = nalum_layer
elif args.nalu == "naluv":
    nalu = naluv_layer
elif args.nalu == "nalu__paperm":
    nalu = nalu_paper_matrix_layer
elif args.nalu == "nalu__paperv":
    nalu = nalu_paper_vector_layer



os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

nalu_architectures.ig = 0
nalu_architectures.im = -1
nalu_architectures.iw = 1

nalu_architectures.isg = 0.1
nalu_architectures.ism = 0.1
nalu_architectures.isw = 0.1


def sample(dist, params, numDim = 2, numDP = 64000):
    data = np.zeros(shape=(numDP, numDim))
    if dist == "normal":
        intmean = (params[0] + params[1]) / 2
        intstd = (params[1] - params[0]) / 6
        print(
            "Generating Data: \nInt: \tdist \t %s\n\t\tdata >=\t %s\n\t\tmean(s)\t %s\n\t\tdata <\t %s\n\t\tstd \t %s" % (
                dist, params[0], intmean, params[1], intstd))
        mi, ma = (params[0] - intmean) / intstd, (params[1] - intmean) / intstd
        data = np.reshape(truncnorm.rvs(mi, ma, intmean, intstd, size=numDim * numDP), data.shape)

    elif dist == "uniform":
        print("Generating Data: \nInt: \tdist \t %s\n\t\tdata >=\t %s\n\t\tdata <\t %s\n\t\t" % (
        dist, params[0], params[1]))
        data = np.reshape(np.random.uniform(params[0], params[1], size=numDim * numDP), data.shape)
    elif dist == "exponential":
        data = np.random.exponential(params, size=(numDP, numDim))
    else:
        raise Exception("Unknown distribution")
    data = np.reshape(data, [-1])  # reshape to mix both distributions per instance!
    np.random.shuffle(data)
    data = np.reshape(data, (numDP, numDim))
    return data




def operation(op, a, b):
    if op.lower() == "mul":
        return a * b
    if op.lower() == "add":
        return a + b
    if op.lower() == "sub":
        return a - b
    if op.lower() == "div":
        return a / b

batch_size = 64
REGULARIZATION = True

random.seed(args.seed)
tf.set_random_seed(args.seed)
np.random.seed(args.seed)

X = tf.placeholder("float", [None, 2])
Y = tf.placeholder("float", [None, 1])

data = sample(args.dist, args.params)
lbls = operation(args.op, data[:,0], data[:,1])
lbls = np.reshape(lbls, newshape=(-1, 1))



int_data = sample(args.dist, args.params)
int_lbls = operation(args.op, int_data[:,0], int_data[:,1])
int_lbls = np.reshape(int_lbls, newshape=(-1, 1))



ext_data = sample(args.dist, args.ext)
ext_lbls = operation(args.op, ext_data[:,0], ext_data[:,1])
ext_lbls = np.reshape(ext_lbls, newshape=(-1, 1))


hidden1, g1, what1, mhat1 = nalu(X, 2, 1)


reg_vars = []
w_vars = []
g_vars = []


for variable in tf.trainable_variables():
    if "what" in variable.name:
        reg_vars.append(variable)
        w_vars.append(variable)
    elif "mhat" in variable.name:
        reg_vars.append(variable)
        w_vars.append(variable)
    elif "g" in variable.name:
        reg_vars.append(variable)
        g_vars.append(variable)
    else:
        print(variable.name)

def regularize(var):
        return 1 * tf.maximum(tf.minimum(-var, var) + 20, 0) # /\ reg. avoid -10 to 10



reg_loss =  0.05 * tf.add_n([tf.reduce_mean(regularize(v)) for v in reg_vars])

loss = tf.losses.mean_squared_error(Y, hidden1)

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)


w_grads = optimizer.compute_gradients(loss, var_list=w_vars)
w_grads = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in w_grads]
train_op_w = optimizer.apply_gradients(w_grads)

g_grads = optimizer.compute_gradients(loss, var_list=g_vars)
g_grads = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in g_grads]
train_op_g = optimizer.apply_gradients(g_grads)

w_reg_grads = optimizer.compute_gradients(reg_loss, var_list=w_vars)
w_reg_grads = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in w_reg_grads]
train_op_reg_w = optimizer.apply_gradients(w_reg_grads)

g_reg_grads = optimizer.compute_gradients(reg_loss, var_list=g_vars)
g_reg_grads = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in g_reg_grads]
train_op_reg_g = optimizer.apply_gradients(g_reg_grads)

all_grads = optimizer.compute_gradients(loss + reg_loss, var_list=tf.trainable_variables())
all_grads = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in all_grads]
train_op_reg = optimizer.apply_gradients(all_grads)


all_grads = optimizer.compute_gradients(loss, var_list=tf.trainable_variables())
all_grads = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in all_grads]
train_op = optimizer.apply_gradients(all_grads)


sess = tf.Session()

def reinit(sess):
    print("REINIT")
    for variable in tf.trainable_variables():
        sess.run(variable.assign(np.random.uniform(-2,2, variable.shape)))

init = tf.global_variables_initializer()
with open("%s.csv" % "_".join([args.output, args.nalu, args.dist, str(args.params), str(args.ext), str(args.seed), args.op, "int"]), "w") as intlog:
    with open("%s.csv" % "_".join([args.output, args.nalu, args.dist, str(args.params), str(args.ext), str(args.seed), args.op, "ext"]), "w") as extlog:

        losses = []
        reinitctr = 0
        with tf.Session() as sess:

            sess.run(init)
            for epoch in range(100):
                i = 0
                print("Epoch:",epoch)
                while i < len(data):
                    X_training_batches = data[i:i + batch_size, :]
                    Y_train_batches = lbls[i:i + batch_size, :]

                    if (i // batch_size) % 10 < 8: # Train w and g seperately
                        eresults, eloss, erloss, _= sess.run([hidden1, loss, reg_loss, train_op_w], feed_dict={X: X_training_batches, Y: Y_train_batches})
                    else:
                        eresults, eloss, erloss, _ = sess.run([hidden1, loss, reg_loss, train_op_g], feed_dict={X: X_training_batches, Y: Y_train_batches})

                    if REGULARIZATION:
                        if epoch > 10 and eloss < 1:
                                eresults, eloss, erloss, _ = sess.run([hidden1, loss, reg_loss, train_op_reg_w], feed_dict={X: X_training_batches, Y: Y_train_batches})
                                eresults, eloss, erloss, _ = sess.run([hidden1, loss, reg_loss, train_op_reg_g], feed_dict={X: X_training_batches, Y: Y_train_batches})

                    losses.append(eloss)
                    if len(losses) > 10000 and epoch > 0 and epoch % 10 == 0 and np.mean(losses[0:len(losses)//2]) <= np.mean(losses[len(losses)//2:]) + np.std(losses[len(losses)//2:]) and np.mean(losses[len(losses)//2:]) > 1:
                        # reinitialization strategy
                        reinit(sess)
                        losses = []
                        reinitctr += 1

                    i += batch_size

                    # check parameters every now and then
                    if (i // batch_size) % 1000 == 0:

                        print("loss: {:.5E}\tregularization-loss: {:.5E}".format(eloss, erloss))
                        print("g:",g1.eval(feed_dict={X: X_training_batches, Y: Y_train_batches}))
                        print("w1:", (tf.tanh(what1) * tf.sigmoid(mhat1)).eval())
                        print(epoch, i,"results",hidden1.eval(feed_dict={X: X_training_batches, Y: Y_train_batches})[0], Y_train_batches[0],)



                        if epoch > 50:
                            good = 0
                            for j, k in enumerate(np.concatenate([Y_train_batches,eresults], axis=1)):
                                # success criterion 1e-5
                                if abs(k[0] - k[1]) < 1e-5:
                                    good += 1
                            print("good: ", good)

                        eresults_ex, eloss_ex, erloss_ex = sess.run([hidden1, loss, reg_loss],
                                                              feed_dict={X: ext_data, Y: ext_lbls})

                        print("ext loss: {:.5E}\tregularization-loss: {:.5E}".format(eloss_ex, erloss_ex))

                        eresults_in, eloss_in, erloss_in = sess.run([hidden1, loss, reg_loss],
                                                              feed_dict={X: int_data, Y: int_lbls})

                        print("int loss: {:.5E}\tregularization-loss: {:.5E}".format(eloss_in, erloss_in))
                        intlog.write("\t".join([str(epoch), str(i), args.output, args.nalu, args.dist, str(args.params), str(args.ext), str(args.seed), args.op, "{:.5E}".format(eloss), "{:.5E}".format(eloss_in), str(reinitctr)])+"\n")
                        extlog.write("\t".join([str(epoch), str(i), args.output, args.nalu, args.dist, str(args.params), str(args.ext), str(args.seed), args.op, "{:.5E}".format(eloss), "{:.5E}".format(eloss_ex), str(reinitctr)]) + "\n")

