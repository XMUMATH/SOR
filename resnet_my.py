This is the resnet structure
'''
import numpy as np
from hyper_parameters import *


BN_EPSILON = 0.001

def _relu(x, leaky=0.2):
    #return 1.0570*(tf.maximum(0.000 , x)+1.6733*(tf.exp(tf.minimum(0.000 , x))-1.000)) # self-normalizing nerual network
    return tf.maximum(leaky*x, x)


def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, wl = 0.0001, my_loss = True, is_fc_layer = False, is_fc_b = False, is_first_conv = False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''
    
    ## TODO: to allow different weight decay to fully connected layer and conv layer
    if is_fc_layer is True:
        var = tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/(shape[0]))))
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name = 'fc_weigth_loss')
    elif is_fc_b is True:
        var = tf.get_variable(name, shape, initializer=tf.constant_initializer(0))
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name = 'fc_bias_loss')
    elif is_first_conv is True:
    	var = tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/(shape[0]*shape[1]*shape[2]))))
    	weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name = 'weigth_loss')
    else:
        var = tf.get_variable(name, shape, initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/(shape[0]*shape[1]*shape[2]))))
        if my_loss is True:
            back_w = tf.transpose(var, perm=[0,1,3,2])
            for_w = var
            for_m = for_w.get_shape()[3].value
            back_m = back_w.get_shape()[3].value 
            reshaped_for_w = tf.reshape(var, [-1,for_m])
            reshaped_back_w = tf.reshape(var, [-1,back_m])
            for_n = reshaped_for_w.get_shape()[0].value
            back_n = reshaped_back_w.get_shape()[0].value
            sub_for_n = for_n//2
            sub_back_n = back_n//2
            pos_for_I = tf.diag(tf.ones([sub_for_n]))
            pos_back_I = tf.diag(tf.ones([sub_back_n]))
            neg_for_I = tf.reshape(tf.image.flip_left_right(tf.reshape(-pos_for_I, [sub_for_n,sub_for_n,1])),[sub_for_n,sub_for_n])
            neg_back_I = tf.reshape(tf.image.flip_left_right(tf.reshape(-pos_back_I, [sub_back_n,sub_back_n,1])),[sub_back_n,sub_back_n])
            block_for_I = tf.concat([tf.concat([pos_for_I,neg_for_I],1), tf.concat([neg_for_I,pos_for_I], 1)],0)
            block_back_I = tf.concat([tf.concat([pos_back_I,neg_back_I],1), tf.concat([neg_back_I,pos_back_I], 1)],0)  #consider about the iso-symmetric structure of w, we here proposed the structured indentity matirx block_I
            for_s = np.sqrt(0.3)*(for_m/for_n)*(2.0/(1.0+0.2)**2)
            back_s = (back_m/back_n)*(1.0/2.0*(1.0+0.2**2)) # this s is corresponding coefficient for block_I.
            regular1 = tf.nn.l2_loss(tf.subtract(tf.matmul(reshaped_for_w, reshaped_for_w, transpose_a=False, transpose_b=True), for_s*block_for_I), name = 'x_weigth_loss_for')\
            +tf.nn.l2_loss(tf.subtract(tf.matmul(reshaped_back_w, reshaped_back_w, transpose_a=False, transpose_b=True), back_s*block_back_I), name = 'x_weigth_loss_back')#+0.3*tf.nn.l2_loss(var)
            weight_loss = tf.multiply(regular1, wl, name = 'weigth_loss')
        else:
            weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name = 'weigth_loss')
    
    tf.add_to_collection('losses', weight_loss)
    return var


def output_layer(input_layer, num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer = True)
    fc_b = create_variables(name='fc_bias', shape=[num_labels], is_fc_b = True)
    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def batch_normalization_layer(input_layer, dimension, batch_wise = False):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    batch_size =  input_layer.get_shape().as_list()[0]
    
    if batch_wise is True:

        mean, variance = tf.nn.moments(input_layer, axes=[1, 2, 3])
        beta = tf.get_variable('beta', batch_size, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', batch_size, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.transpose(tf.nn.batch_normalization(tf.transpose(input_layer, perm=[3,1,2,0]), mean, variance, beta, gamma, BN_EPSILON), perm=[3,1,2,0])

    else:

        mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
        beta = tf.get_variable('beta', dimension, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))
        gamma = tf.get_variable('gamma', dimension, tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
        bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)

    return bn_layer






def conv_bn_relu_layer(input_layer, filter_shape, stride, BN = False, first_conv = False):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape, is_first_conv = first_conv)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    if BN is True:
        output = _relu(batch_normalization_layer(conv_layer, out_channel))
    else:
        output = _relu(conv_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride, BN = False):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''
    in_channel = input_layer.get_shape().as_list()[-1]
    if BN is True:          
        relu_layer = _relu(batch_normalization_layer(input_layer, in_channel))
    else:
        relu_layer = _relu(input_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer



def residual_block(input_layer, output_channel, first_block = False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    #if increase_dim is True:
    #    projection = create_variables(name='projection', shape=[1, 1, input_channel, output_channel], wl=0.0)
    #    input_ = tf.nn.conv2d(input_layer, filter=projection, strides=[1, 2, 2, 1], padding='SAME')
    #else:
    #   input_ = input_layer
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer
        
    output = conv2 #+ padded_input
    #output = batch_normalization_layer(output, output.get_shape().as_list()[-1], batch_wise = True)
    return output


def inference(input_tensor_batch, n, reuse):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''

    layers = []
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1, first_conv = True)
        activation_summary(conv0)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('conv1_%d' %i, reuse=reuse):
            if i == 0:
                conv1 = residual_block(layers[-1], 16, first_block=True)
            else:
                conv1 = residual_block(layers[-1], 16)
            activation_summary(conv1)
            layers.append(conv1)

    for i in range(n):
        with tf.variable_scope('conv2_%d' %i, reuse=reuse):
            conv2 = residual_block(layers[-1], 32)
            activation_summary(conv2)
            layers.append(conv2)

    for i in range(n):
        with tf.variable_scope('conv3_%d' %i, reuse=reuse):
            conv3 = residual_block(layers[-1], 64)
            layers.append(conv3)
        assert conv3.get_shape().as_list()[1:] == [8, 8, 64]

    with tf.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        #bn_layer = batch_normalization_layer(layers[-1], in_channel)
        #bn_layer = batch_normalization_layer(layers[-1], in_channel, batch_wise = False)
        #relu_layer = _relu(bn_layer)
        relu_layer = _relu(layers[-1])
        global_pool = tf.reduce_mean(relu_layer, [1, 2])

        assert global_pool.get_shape().as_list()[-1:] == [64]
        output = output_layer(global_pool, 10)
        
        layers.append(output)

    return layers[-1]


def test_graph(train_dir='logs'):
    '''
    Run this function to look at the graph structure on tensorboard. A fast way!
    :param train_dir:
    '''
    input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
    result = inference(input_tensor, 2, reuse=False)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
