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
from scipy.stats import truncnorm
from nalu_architectures import *


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", dest="output", default="naly_syn_simple_func")

parser.add_argument("-n", "--nalu", dest="nalu", default="nalui2")
parser.add_argument("-se", "--seed", dest="seed", default=42, type=int)
parser.add_argument("-op", "--operation", dest="op", default="SUB")

parser.add_argument("-d", "--dist", dest="dist", default="uniform", help="Prob.Dist")
parser.add_argument("-p", "--params",dest="params" , default="(-3,3)", type=ast.literal_eval)
parser.add_argument("-e", "--ext",dest="ext" , default="(3,5)", type=ast.literal_eval)



args = parser.parse_args()



def gen_data(numDP, extrapolate=False, operation=np.divide, abselector=None):

    assert(args.ext[0] < args.ext[1])
    assert(args.params[0] < args.params[1])

    if args.ext[0] <= args.params[0]:
        assert(args.ext[1] <= args.params[0])
    else:
        assert(args.ext[0] >= args.params[1])


    numDim = 100
    data = np.zeros(shape=(numDP, numDim))
    labels = np.zeros(shape=(numDP, 1))

    if abselector is None:
        abselector = np.round(np.random.uniform(0, 2, numDim)) - 1

    la, lb = [], []

    if extrapolate == False:
        if args.dist == "normal":
            intmean = (args.params[0] + args.params[1]) / 2
            intstd = (args.params[1] - args.params[0]) / 6
            print("Generating Data: \nInt: \tdist \t %s\n\t\tdata >=\t %s\n\t\tmean(s)\t %s\n\t\tdata <\t %s\n\t\tstd \t %s" % (
                    args.dist, args.params[0], intmean, args.params[1], intstd))
            mi, ma = (args.params[0] - intmean) / intstd, (args.params[1] - intmean) / intstd
            data = np.reshape(truncnorm.rvs(mi, ma, intmean, intstd, size=numDim * numDP), data.shape)

        elif args.dist == "uniform":
            print("Generating Data: \nInt: \tdist \t %s\n\t\tdata >=\t %s\n\t\tdata <\t %s\n\t\t" % (args.dist, args.params[0],args.params[1]))
            data = np.reshape(np.random.uniform(args.params[0], args.params[1], size=numDim * numDP), data.shape)

    else:

        data = np.zeros(shape=(numDP, numDim))
        if args.dist == "normal":
            if type(args.ext) in [list, tuple]:
                extmean = (args.ext[0] + args.ext[1])/2
                extstd = (args.ext[1] - args.ext[0])/6
                print(
                    "Generating Data: \nExt: \tdist \t %s\n\t\tdata >=\t %s\n\t\tmean(s)\t %s\n\t\tdata <\t %s\n\t\tstd \t %s" % (
                        args.dist, args.ext[0], extmean, args.ext[1], extstd))
                mi, ma = (args.ext[0] - extmean) / extstd,  (args.ext[1] - extmean) / extstd
                data = np.reshape(truncnorm.rvs(mi, ma, extmean, extstd, size=len(data.reshape([-1]))), data.shape)
            else:
                raise Exception("don't know what to do")
        elif args.dist == "uniform":
            print("Generating Data: \nExt: \tdist \t %s\n\t\tdata >=\t %s\n\t\tdata <\t %s\n\t\t" % (
            args.dist, args.ext[0], args.ext[1]))
            data = np.reshape(np.random.uniform(args.ext[0],args.ext[1],
                                   size=len(data.reshape([-1]))), data.shape)

    data = np.reshape(data, [-1]) # reshape to mix both distributions per instance!
    np.random.shuffle(data)
    data = np.reshape(data, (numDP, numDim))    # reshape mixed dist to original shape


    for i in range(0, numDP):
        a = 0
        b = 0

        for j in range(0, numDim):

            if abselector[j] == 0:
                a += data[i][j]
            elif abselector[j] == 1:    # if we generate ie -1, 0, 1 we can introduce input variables which are totally unrelated for the calculation
                b += data[i][j]

        la.append(a)
        lb.append(b)
        labels[i][0] = operation(a, b)

    return data, labels, abselector, np.array(la), np.array(lb)


def datastats(data, labels, absel, la, lb, extrapol, op, text=None):
    import matplotlib
    matplotlib.use('Agg')

    from matplotlib import pyplot as plt

    plt.figure(figsize=(10,10))

    ax1 = plt.subplot(511)
    plt.hist(data.reshape([-1]), bins=100)
    ax1.set_title("Data distribution")
    ax2 = plt.subplot(512)
    plt.hist(la.reshape([-1]), bins=100)
    ax2.set_title("Distribution of a")
    ax3 = plt.subplot(513, sharex=ax2)
    plt.hist(lb.reshape([-1]), bins=100)
    ax3.set_title("Distribution of b")
    ax4 = plt.subplot(514)
    plt.hist(np.array(labels).reshape([-1]), bins=100)
    ax4.set_title("Result distribution a %s b" % op)
    ax5 = plt.subplot(515)
    plt.imshow(np.reshape(absel, [1, -1]), cmap="Greys") # off (-1): white          a (0): grey         b (1) black
    ax5.set_title("a/b selector")
    ax2.set_xlim([-200,200])
    plt.axis('off')

    plt.suptitle('Generated Data' + (" (%s)" % text if text is not None else ""), fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    return plt




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


if args.op.lower() == "mul":
    op = np.multiply
if args.op.lower() == "add":
    op = np.add
if args.op.lower() == "sub":
    op = np.subtract
if args.op.lower() == "div":
    op = np.divide

numDPs = 64000

data, lbls, abselector, la, lb = gen_data(numDPs, extrapolate=False, operation=op)
int_data, int_lbls, abselector_int, la_int, lb_int = gen_data(64000, extrapolate=False,
                                                                operation=op, abselector=abselector)
ext_data, ext_lbls, abselector_ext, la_ext, lb_ext = gen_data(64000, extrapolate=True,
                                                                operation=op, abselector=abselector)


batch_size = 64
REGULARIZATION = True

random.seed(args.seed)
tf.set_random_seed(args.seed)
np.random.seed(args.seed)

X = tf.placeholder("float", [None, 100])
Y = tf.placeholder("float", [None, 1])



logits, g1, g2, what1, mhat1, what2, mhat2, hidden1 = nalu_net(X, 100, 2, 1, nalu_layer=nalu)

print("Interpolation Data and Labels:")
print(int_data[0:3])
print(int_lbls[:3])

print("Extrapolation Data and Labels:")
print(ext_data[0:3])
print(ext_lbls[:3])

print("Training Data and Labels:")
print(data[0:3])
print(lbls[:3])


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

loss = tf.losses.mean_squared_error(Y, logits)

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
                        eresults, eloss, erloss, _= sess.run([logits, loss, reg_loss, train_op_w], feed_dict={X: X_training_batches, Y: Y_train_batches})
                    else:
                        eresults, eloss, erloss, _ = sess.run([logits, loss, reg_loss, train_op_g], feed_dict={X: X_training_batches, Y: Y_train_batches})
                    if REGULARIZATION:
                        if epoch > 10 and eloss < 1:
                                eresults, eloss, erloss, _ = sess.run([logits, loss, reg_loss, train_op_reg_w], feed_dict={X: X_training_batches, Y: Y_train_batches})
                                eresults, eloss, erloss, _ = sess.run([logits, loss, reg_loss, train_op_reg_g], feed_dict={X: X_training_batches, Y: Y_train_batches})

                    losses.append(eloss)
                    if len(losses) > 10000 and epoch > 0 and epoch % 10 == 0 and np.mean(losses[0:len(losses)//2]) <= np.mean(losses[len(losses)//2:]) + np.std(losses[len(losses)//2:]) and np.mean(losses[len(losses)//2:]) > 1:
                        # reinitialization strategy
                        reinit(sess)
                        losses = []
                        reinitctr += 1


                    i += batch_size

                    # check parameters every now and then
                    if (i // batch_size) % 1000 == 0:
                        print("g:",g1.eval(feed_dict={X: X_training_batches, Y: Y_train_batches}))
                        print("w:", (tf.tanh(what1) * tf.sigmoid(mhat1)).eval())
                        print("g2:", g2.eval(feed_dict={X: X_training_batches, Y: Y_train_batches}))
                        print("w2:", (tf.tanh(what2) * tf.sigmoid(mhat2)).eval())
                        print(epoch, i,"results",logits.eval(feed_dict={X: X_training_batches, Y: Y_train_batches})[0], Y_train_batches[0],)



                        if epoch > 50:
                            good = 0
                            for j, k in enumerate(np.concatenate([Y_train_batches,eresults], axis=1)):
                                # success criterion 1e-5
                                if abs(k[0] - k[1]) < 1e-5:
                                    good += 1
                            print("good: ", good)

                        eresults_ex, eloss_ex, erloss_ex, ehidden1 = sess.run([logits, loss, reg_loss, hidden1],
                                                              feed_dict={X: ext_data, Y: ext_lbls})
                        print("loss: {:.5E}\tregularization-loss: {:.5E}".format(eloss, erloss))
                        print("input\n",ext_data[0:5],"\nabsel\n",abselector,"\nla\n",la_ext[0:5],"\nlb\n", lb_ext[0:5],"\na1\n","\nhidden1\n",ehidden1[0:5], "\nhidden2\n" ,eresults_ex[0:5], "\ny\n",ext_lbls[0:5])

                        print("ext loss: {:.5E}\tregularization-loss: {:.5E}".format(eloss_ex, erloss_ex))

                        eresults_in, eloss_in, erloss_in = sess.run([logits, loss, reg_loss],
                                                              feed_dict={X: int_data, Y: int_lbls})

                        print("int loss: {:.5E}\tregularization-loss: {:.5E}".format(eloss_in, erloss_in))
                        intlog.write("\t".join([str(epoch), str(i), args.output, args.nalu, args.dist, str(args.params), str(args.ext), str(args.seed), args.op, "{:.5E}".format(eloss), "{:.5E}".format(eloss_in), str(reinitctr)])+"\n")
                        extlog.write("\t".join([str(epoch), str(i), args.output, args.nalu, args.dist, str(args.params), str(args.ext), str(args.seed), args.op, "{:.5E}".format(eloss), "{:.5E}".format(eloss_ex), str(reinitctr)]) + "\n")
