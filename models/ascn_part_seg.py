import tensorflow as tf
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../models'))
import tf_util


def combine_heads(x):
    x = tf.transpose(x, [0, 2, 1, 3])
    old_shape = x.get_shape().dims
    a, b = old_shape[-2:]
    new_shape = old_shape[:-2] + [a * b if a and b else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
    ret.set_shape(new_shape)

    return ret


def split_heads(x, n):
    old_shape = x.get_shape().dims
    # print len(old_shape)
    if len(old_shape) == 2:
        x = tf.expand_dims(x, 0)
    # print x.get_shape()
    old_shape = x.get_shape().dims
    last = old_shape[-1]
    new_shape = old_shape[:-1] + [n] + [last // n if last else None]
    ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
    ret.set_shape(new_shape)
    return tf.transpose(ret, [0, 2, 1, 3])


def transformer(ind, input_tensor,
                dk,
                dv,
                num_heads,
                attention_dropout,
                is_training,
                residual_on=True):
    q = tf_util.conv2d(
        tf.expand_dims(input_tensor, 2),
        dk, [1, 1],
        padding='VALID',
        stride=[1, 1],
        is_training=is_training,
        bn=False,
        activation_fn=None,
        scope=str(ind) + '_transformer_q')
    k = tf_util.conv2d(
        tf.expand_dims(input_tensor, 2),
        dk, [1, 1],
        padding='VALID',
        stride=[1, 1],
        is_training=is_training,
        bn=False,
        activation_fn=None,
        scope=str(ind) + '_transformer_k')
    v = tf_util.conv2d(
        tf.expand_dims(input_tensor, 2),
        dv, [1, 1],
        padding='VALID',
        stride=[1, 1],
        is_training=is_training,
        bn=True,
        activation_fn=None,
        scope=str(ind) + '_transformer_v')
    bias = None

    q = tf.squeeze(q)
    k = tf.squeeze(k)
    v = tf.squeeze(v)
    q = split_heads(q, num_heads)
    k = split_heads(k, num_heads)
    key_depth_per_head = dk // num_heads
    q *= key_depth_per_head ** -0.5

    logits = tf.matmul(q, k, transpose_b=True)

    if bias is not None:
        logits += bias

    weights = tf.nn.softmax(logits,
                            name=str(ind) + '_attention_weights')

    weights = tf_util.dropout(weights,
                              keep_prob=1.0 - attention_dropout,
                              is_training=is_training,
                              scope=str(ind) + '_attn_dp')

    v = split_heads(v, num_heads)
    x = tf.matmul(weights, v)
    x = combine_heads(x)

    output_depth = dv
    x = tf_util.conv2d(
        tf.expand_dims(x, 2),
        output_depth, [1, 1],
        padding='VALID',
        stride=[1, 1],
        is_training=is_training,
        bn=True,
        scope=str(ind) + '_transformer_output')

    x = tf.squeeze(x)

    if residual_on:
        return combine_heads(v) + x
    else:
        return x


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def add_gated_layers(ind, net, is_training, r=4):
    D = int(net.shape[-1])
    netc = tf.expand_dims(net, 2)
    netc = tf_util.conv2d(netc, D, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training,
                          scope=str(ind) + '_conv1')
    netc = tf_util.conv2d(netc, D / r, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training,
                          scope=str(ind) + '_conv2')
    netc = tf_util.conv2d(netc, D, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=True, is_training=is_training,
                          activation_fn=tf.nn.sigmoid,
                          scope=str(ind) + '_conv3')
    netc = tf.squeeze(netc, 2)
    return tf.multiply(net, netc)


def get_model(point_cloud, input_label, is_training, cat_num, part_num,
              weight_decay, num_heads=1, bn_decay=None):
    """ Classification sefl attention net, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    # First transformer + conv2d + gate
    net = transformer(1, point_cloud, 32, 64, num_heads, 0, is_training, residual_on=True)
    net = tf.expand_dims(net, 2)
    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf.squeeze(net, 2)
    net = add_gated_layers(1.5, net, is_training)
    out_1 = tf.expand_dims(net, 2)
    # Second transformer + conv2d + gate
    net = transformer(2, net, 32, 64, num_heads, 0, is_training, residual_on=True)
    net = tf.expand_dims(net, 2)
    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    net = tf.squeeze(net, 2)
    net = add_gated_layers(2.5, net, is_training)
    out_2 = tf.expand_dims(net, 2)

    # 3rd transformer + conv2d + gate
    net = transformer(3, net, 32, 64, num_heads, 0, is_training, residual_on=True)
    net = tf.expand_dims(net, 2)
    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf.squeeze(net, 2)
    net = add_gated_layers(3.5, net, is_training)
    out_3 = tf.expand_dims(net, 2)

    # 4th transformer + covn2d + gate
    net = transformer(4, net, 32, 128, num_heads, 0, is_training, residual_on=True)
    net = tf.expand_dims(net, 2)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf.squeeze(net, 2)
    net = add_gated_layers(4.5, net, is_training)
    out_4 = tf.expand_dims(net, 2)

    # 5th transformer + conv2d + gate
    net = transformer(5, net, 64, 1024, num_heads, 0, is_training, residual_on=True)
    net = tf.expand_dims(net, 2)
    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)
    net = tf.squeeze(net, 2)
    net = add_gated_layers(5.5, net, is_training)
    out_5 = tf.expand_dims(net, 2)

    # Symmetric function: max pooling
    net = tf.expand_dims(net, 2)
    out_max = tf_util.max_pool2d(net, [num_point, 1],
                                 padding='VALID', scope='maxpool')

    # MLP on global point cloud vector
    net = tf.reshape(out_max, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, cat_num, activation_fn=None, scope='fc3')

    # segmentation network
    one_hot_label_expand = tf.reshape(input_label, [batch_size, 1, 1, cat_num])
    out_max = tf.concat(axis=3, values=[out_max, one_hot_label_expand])

    expand = tf.tile(out_max, [1, num_point, 1, 1])
    concat = tf.concat(axis=3, values=[expand, out_1, out_2, out_3, out_4, out_5])

    net2 = tf_util.conv2d(concat, 256, [1, 1], padding='VALID', stride=[1, 1], bn_decay=bn_decay,
                          bn=True, is_training=is_training, scope='seg/conv1', weight_decay=weight_decay)
    net2 = tf_util.dropout(net2, keep_prob=0.8, is_training=is_training, scope='seg/dp1')
    net2 = tf_util.conv2d(net2, 256, [1, 1], padding='VALID', stride=[1, 1], bn_decay=bn_decay,
                          bn=True, is_training=is_training, scope='seg/conv2', weight_decay=weight_decay)
    net2 = tf_util.dropout(net2, keep_prob=0.8, is_training=is_training, scope='seg/dp2')
    net2 = tf_util.conv2d(net2, 128, [1, 1], padding='VALID', stride=[1, 1], bn_decay=bn_decay,
                          bn=True, is_training=is_training, scope='seg/conv3', weight_decay=weight_decay)
    net2 = tf_util.conv2d(net2, part_num, [1, 1], padding='VALID', stride=[1, 1], activation_fn=None,
                          bn=False, scope='seg/conv4', weight_decay=weight_decay)

    net2 = tf.reshape(net2, [batch_size, num_point, part_num])

    return net, net2

def get_loss(l_pred, seg_pred, label, seg, weight):
    per_instance_label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=l_pred, labels=label)
    label_loss = tf.reduce_mean(per_instance_label_loss)

    # size of seg_pred is batch_size x point_num x part_cat_num
    # size of seg is batch_size x point_num
    per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg), axis=1)
    seg_loss = tf.reduce_mean(per_instance_seg_loss)

    per_instance_seg_pred_res = tf.argmax(seg_pred, 2)

    total_loss = weight * seg_loss + (1 - weight) * label_loss

    return total_loss, label_loss, per_instance_label_loss, seg_loss, per_instance_seg_loss, per_instance_seg_pred_res

