import numpy as np
import tensorflow as tf

ig = 0.0
im = 0.5
iw = 0.88

isg = 0.2
ism = 0.2
isw = 0.2



def activation_layer(input, inputdim, outputdim, activation):
    out_w = tf.Variable(initialization([outputdim, inputdim])
    out_b = tf.Variable(initialization([outputdim]))
    out_layer = tf.add(tf.matmul(input, out_w, transpose_b=True), out_b)
    return activation(out_layer)




def softexp(x):
    alpha = tf.Variable(0.0, dtype=tf.float32)
    zero = tf.constant(0.0)
    def gt(x):
        return alpha + (tf.exp(alpha * x) - 1.) / alpha
    def lt(x):
        return - tf.log(1 - alpha * (x + alpha)) / alpha
    return tf.case( pred_fn_pairs=[
        (tf.less(alpha, zero), lambda : lt(x)),
        (tf.greater(alpha, zero), lambda : gt(x))],
        default=lambda : x)


def normal_net(x):
    return relu_net(x, activation=lambda x: x)


def relu_net(x, num_input, n_hidden_1, n_hidden_2, activation=tf.nn.relu):
    with tf.variable_scope("layer1"):
        hidden1 = activation_layer(x, num_input, n_hidden_1, activation=activation)
    with tf.variable_scope("layer2"):
        hidden2 = activation_layer(hidden1, n_hidden_1, n_hidden_2, activation=activation)
    return hidden2


def initialization(dimension,
                   activationrange=2):  # glorot initialization. mean and stddev for backwards compatibility with truncated normal
    if type(dimension) is list and len(dimension) > 1:
        input_dim = dimension[0]
        output_dim = dimension[1]
    elif type(dimension) is list:
        input_dim = 0
        output_dim = dimension[0]
    else:
        input_dim = 0
        output_dim = dimension
    return np.random.uniform(
        low=-np.sqrt(activationrange / (output_dim + input_dim)),
        high=np.sqrt(activationrange / (output_dim + input_dim)),
        size=dimension
    ).astype('float32')


def g_initialization(dimension,
                     activationrange=2):  # glorot initialization. mean and stddev for backwards compatibility with truncated normal
    if type(dimension) is list and len(dimension) > 1:
        input_dim = dimension[0]
        output_dim = dimension[1]
    elif type(dimension) is list:
        input_dim = 0
        output_dim = dimension[0]
    else:
        input_dim = 0
        output_dim = dimension
    return np.random.normal(loc=ig, scale=isg, size=dimension).astype('float32')


def w_initialization(dimension,
                     activationrange=2):  # glorot initialization. mean and stddev for backwards compatibility with truncated normal
    if type(dimension) is list and len(dimension) > 1:
        input_dim = dimension[0]
        output_dim = dimension[1]
    elif type(dimension) is list:
        input_dim = 0
        output_dim = dimension[0]
    else:
        input_dim = 0
        output_dim = dimension
    return np.random.normal(loc=iw, scale=isw, size=dimension).astype('float32')


def m_initialization(dimension,
                     activationrange=2):  # glorot initialization. mean and stddev for backwards compatibility with truncated normal
    if type(dimension) is list and len(dimension) > 1:
        input_dim = dimension[0]
        output_dim = dimension[1]
    elif type(dimension) is list:
        input_dim = 0
        output_dim = dimension[0]
    else:
        input_dim = 0
        output_dim = dimension

    return np.random.normal(loc=im, scale=ism, size=dimension).astype('float32')


# NALU Layerss

def nalu_paper_matrix_layer(input, inputdim, outputdim, force="none", eps=1e-7):  #nalu paper matrix
    """
    NALU Layer as often implemented. Vector (wight matrix x input) as gate, no clipping of MUL
    :param input:
    :param inputdim:
    :param outputdim:
    :return:
    """

    w_hat1 = tf.Variable(w_initialization([inputdim, outputdim]), name="what")
    m_hat1 = tf.Variable(m_initialization([inputdim, outputdim]), name="what")


    G1 = tf.Variable(g_initialization([inputdim, outputdim]), name="g")
    W1 = tf.tanh(w_hat1) * tf.sigmoid(m_hat1)
    a1 = tf.matmul(input, W1)
    g1 = tf.sigmoid(tf.matmul(input, G1))

    m1 = tf.exp(tf.matmul(tf.log(tf.abs(input) + eps), W1))  # [1, 20]


    if force == "add":
        print("FORCING ADD")
        out = a1
    elif force == "mul":
        print("FORCING MUL")
        out = m1
    else:
        out = g1 * a1 + (1 - g1) * m1

    return out, g1, w_hat1, m_hat1 



def nalu_paper_vector_layer(input, inputdim, outputdim):
    """
    NALU Layer as described in the paper. Scalar (wight vector x input) as gate, no clipping of MUL
    :param input:
    :param inputdim:
    :param outputdim:
    :return:
    """
    w_hat1 = tf.Variable(w_initialization([outputdim, inputdim]), name="what")
    m_hat1 = tf.Variable(m_initialization([outputdim, inputdim]), name="mhat")
    G1 = tf.Variable(g_initialization(inputdim), name="g")

    G1 = tf.reshape(G1, [inputdim, -1])
    W1 = tf.tanh(w_hat1) * tf.sigmoid(m_hat1)
    a1 = tf.matmul(input, W1, transpose_b=True)
    g1 = tf.sigmoid(tf.matmul(input, G1))
    m0 = tf.exp(tf.matmul(tf.log(tf.abs(input) + 1e-7), W1, transpose_b=True))


    out = g1 * a1 + (1 - g1) * m0

    return out, g1, w_hat1, m_hat1


def naluv_layer(input, inputdim, outputdim):
    """
    NALU Layer as described in the paper. Scalar (wight vector x input) as gate plus MUL clipping.
    :param input:
    :param inputdim:
    :param outputdim:
    :return:
    """
    w_hat1 = tf.Variable(w_initialization([outputdim, inputdim]), name="what")
    m_hat1 = tf.Variable(m_initialization([outputdim, inputdim]), name="mhat")
    G1 = tf.Variable(g_initialization(inputdim), name="g")
    G1 = tf.reshape(G1, [inputdim, -1])
    W1 = tf.tanh(w_hat1) * tf.sigmoid(m_hat1)
    W1 = tf.transpose(W1)
    a1 = tf.matmul(input, W1)
    g1 = tf.sigmoid(tf.matmul(tf.abs(input), G1))
    m1 = tf.exp(tf.minimum(tf.matmul(tf.log(tf.abs(input) + 1e-7), W1), 20)) # clipping

    ### sign

    W1s = tf.reshape(W1, [-1]) # flatten W1s to (200)
    W1s = tf.abs(W1s)
    Xs = tf.concat([input] * W1.shape[1], axis=1)
    Xs = tf.reshape(Xs, shape=[-1,W1.shape[0] * W1.shape[1]])
    sgn = tf.sign(Xs) * W1s + (1 - W1s)
    sgn = tf.reshape(sgn, shape=[-1, W1.shape[1], W1.shape[0]])
    ms1 = tf.reduce_prod(sgn, axis=2)

    out = g1 * a1 + (1 - g1) * m1 * tf.clip_by_value(ms1, -1, 1)

    return out, g1, w_hat1, m_hat1


def nalui2_layer(input, inputdim, outputdim):
    """
    NALU Layer with seperate weights without input dependence of gates. Scalar (wight vector) as gate plus MUL clipping.
    :param input:
    :param inputdim:
    :param outputdim:
    :return:
    """

    w_hat1 = tf.Variable(w_initialization([inputdim, outputdim]), name="what")
    m_hat1 = tf.Variable(m_initialization([inputdim, outputdim]), name="mhat")
    w_hat2 = tf.Variable(w_initialization([inputdim, outputdim]), name="whatm")
    m_hat2 = tf.Variable(m_initialization([inputdim, outputdim]), name="mhatm")


    W1 = tf.tanh(w_hat1) * tf.sigmoid(m_hat1)
    W2 = tf.tanh(w_hat2) * tf.sigmoid(m_hat2)
    a1 = tf.matmul(input, W1)

    m1 = tf.exp(tf.minimum(tf.matmul(tf.log(tf.maximum(tf.abs(input), 1e-7)), W2), 20)) # clipping

    ### sign

    W1s = tf.reshape(W2, [-1]) # flatten W1s to (200)
    W1s = tf.abs(W1s)
    Xs = tf.concat([input] * W1.shape[1], axis=1)
    Xs = tf.reshape(Xs, shape=[-1,W1.shape[0] * W1.shape[1]])
    sgn = tf.sign(Xs) * W1s + (1 - W1s)
    sgn = tf.reshape(sgn, shape=[-1, W1.shape[1], W1.shape[0]])
    ms1 = tf.reduce_prod(sgn, axis=2)


    G1 = tf.Variable(g_initialization(outputdim), name="g") 
    g1 = tf.sigmoid(G1) 
    out = g1 * a1 + (1 - g1) * m1 * tf.clip_by_value(ms1, -1, 1)

    return out, g1, w_hat1, m_hat1 


def nalui1_layer(input, inputdim, outputdim):
    """
    NALU Layer with shared weights without input dependence of gates. Scalar (wight vector) as gate plus MUL clipping.
    :param input:
    :param inputdim:
    :param outputdim:
    :return:
    """

    w_hat1 = tf.Variable(w_initialization([inputdim, outputdim]), name="what")
    m_hat1 = tf.Variable(m_initialization([inputdim, outputdim]), name="mhat")


    W1 = tf.tanh(w_hat1) * tf.sigmoid(m_hat1)
    a1 = tf.matmul(input, W1)

    m1 = tf.exp(tf.minimum(tf.matmul(tf.log(tf.maximum(tf.abs(input), 1e-7)), W1), 20))

    ### sign

    W1s = tf.reshape(W1, [-1]) # flatten W1s to (200)
    W1s = tf.abs(W1s)
    Xs = tf.concat([input] * W1.shape[1], axis=1)
    Xs = tf.reshape(Xs, shape=[-1,W1.shape[0] * W1.shape[1]])
    sgn = tf.sign(Xs) * W1s + (1 - W1s)
    sgn = tf.reshape(sgn, shape=[-1, W1.shape[1], W1.shape[0]])
    ms1 = tf.reduce_prod(sgn, axis=2)


    G1 = tf.Variable(g_initialization(outputdim), name="g")
    g1 = tf.sigmoid(G1)
    out = g1 * a1 + (1 - g1) * m1 * tf.clip_by_value(ms1, -1, 1)

    return out, g1, w_hat1, m_hat1



def nalum_layer(input, inputdim, outputdim, force="none", eps=1e-7): #1e-7
    """
    NALU Layer with matrix weights. Vector (wight matrix x input) as gate plus MUL clipping and sign.
    :param input:
    :param inputdim:
    :param outputdim:
    :return:
    """

    w_hat1 = tf.Variable(w_initialization([inputdim, outputdim]), name="what") 
    m_hat1 = tf.Variable(m_initialization([inputdim, outputdim]), name="what")


    G1 = tf.Variable(g_initialization([inputdim, outputdim]), name="g")
    W1 = tf.tanh(w_hat1) * tf.sigmoid(m_hat1) 
    a1 = tf.matmul(input, W1)
    g1 = tf.sigmoid(tf.matmul(tf.abs(input), G1))


    m1 = tf.exp(tf.minimum(tf.matmul(tf.log(tf.maximum(tf.abs(input), eps)), W1), 20))


    ### sign


    W1s = tf.reshape(W1, [-1]) # flatten W1s to (200)
    W1s = tf.abs(W1s)
    Xs = tf.concat([input] * W1.shape[1], axis=1)
    Xs = tf.reshape(Xs, shape=[-1,W1.shape[0] * W1.shape[1]])
    sgn = tf.sign(Xs) * W1s + (1 - W1s)
    sgn = tf.reshape(sgn, shape=[-1, W1.shape[1], W1.shape[0]])
    ms1 = tf.reduce_prod(sgn, axis=2)




    if force == "add":
        print("FORCING ADD")
        out = a1
    elif force == "mul":
        print("FORCING MUL")
        out = m1
    else:
        out = g1 * a1 + (1 - g1) * m1 * tf.clip_by_value(ms1, -1, 1)


    return out, g1, w_hat1, m_hat1 


# Create NALU model
def nalu_net(x, num_input, n_hidden_1, n_hidden_2, nalu_layer=naluv_layer):  # [16,1]

    # Hidden Layer NALU
    #  Defining Neural Arithmetic Logic unit - 1 and 2
    with tf.variable_scope("layer1"):
        hidden1, g1, what1, mhat1 = nalu_layer(x, num_input, n_hidden_1)
    with tf.variable_scope("layer2"):
        hidden2, g2, what2, mhat2 = nalu_layer(hidden1, n_hidden_1, n_hidden_2)

    return hidden2, g1, g2, what1, mhat1, what2, mhat2, hidden1

def nalu_deep_net(x, num_input, n_hidden_1, n_hidden_2, nalu_layer=naluv_layer):  # [16,1]

    # Hidden Layer NALU
    #  Defining Neural Arithmetic Logic unit - 1 to 6
    with tf.variable_scope("layer1"):
        hidden1, g1, what1, mhat1 = nalu_layer(x, num_input, num_input)
    with tf.variable_scope("layer2"):
        hidden2, g2, what2, mhat2 = nalu_layer(hidden1, num_input, num_input)
    with tf.variable_scope("layer3"):
        hidden3, _, _, _ = nalu_layer(hidden2, num_input, num_input)
    with tf.variable_scope("layer4"):
        hidden4, _, _, _ = nalu_layer(hidden3, num_input, num_input)
    with tf.variable_scope("layer5"):
        hidden5, _, _, _ = nalu_layer(hidden4, num_input, n_hidden_1)
    with tf.variable_scope("layer6"):
        hidden6, _, _, _ = nalu_layer(hidden5, n_hidden_1, n_hidden_2)

    return hidden2, g1, g2, what1, mhat1, what2, mhat2
