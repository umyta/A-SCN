import tensorflow as tf
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
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
        bn=True,
        activation_fn=None,
        scope=str(ind) + '_transformer_q')
    k = tf_util.conv2d(
        tf.expand_dims(input_tensor, 2),
        dk, [1, 1],
        padding='VALID',
        stride=[1, 1],
        is_training=is_training,
        bn=True,
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

    q = tf.squeeze(q, 2)
    k = tf.squeeze(k, 2)
    v = tf.squeeze(v, 2)
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


def get_model(point_cloud, is_training, num_heads=4, bn_decay=None, adrop=0):
    """ Classification sefl attention net, input is BxNx3, output Bx40 """
    model = ASCN(point_cloud, is_training, num_heads, bn_decay, adrop)
    return model.get_model()


class ASCN():
    def __init__(self, point_cloud, is_training, num_heads=1, bn_decay=None, adrop=0.1):
        self.point_cloud = point_cloud
        self.is_training = is_training
        self.num_heads = num_heads
        self.bn_decay = bn_decay
        self.adrop = adrop

    def get_model(self):
        batch_size = self.point_cloud.get_shape()[0].value
        num_point = self.point_cloud.get_shape()[1].value

        end_points = {}

        input_image = self.point_cloud
        # First transformer + conv2d + gate
        net = transformer(1, input_image, 32, 64, self.num_heads, self.adrop, self.is_training, residual_on=True)
        net = tf.expand_dims(net, 2)
        net = tf_util.conv2d(net, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training,
                             scope='conv1', bn_decay=self.bn_decay)
        net = tf.squeeze(net, 2)
        net = add_gated_layers(1.5, net, self.is_training)

        # Second transformer + conv2d + gate
        net = transformer(2, net, 32, 64, self.num_heads, self.adrop, self.is_training, residual_on=True)
        net = tf.expand_dims(net, 2)
        net = tf_util.conv2d(net, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training,
                             scope='conv2', bn_decay=self.bn_decay)
        net = tf.squeeze(net, 2)
        net = add_gated_layers(2.5, net, self.is_training)

        # 3rd transformer + conv2d + gate
        net = transformer(3, net, 32, 64, self.num_heads, self.adrop, self.is_training, residual_on=True)
        net = tf.expand_dims(net, 2)
        net = tf_util.conv2d(net, 64, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training,
                             scope='conv3', bn_decay=self.bn_decay)
        net = tf.squeeze(net, 2)
        net = add_gated_layers(3.5, net, self.is_training)

        net_transformed = net

        # 4th transformer + covn2d + gate
        net = transformer(4, net_transformed, 32, 128, self.num_heads, self.adrop, self.is_training, residual_on=True)
        net = tf.expand_dims(net, 2)
        net = tf_util.conv2d(net, 128, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training,
                             scope='conv4', bn_decay=self.bn_decay)
        net = tf.squeeze(net, 2)
        net = add_gated_layers(4.5, net, self.is_training)

        # 5th transformer + conv2d + gate
        net = transformer(5, net, 64, 1024, self.num_heads, self.adrop, self.is_training, residual_on=True)
        net = tf.expand_dims(net, 2)
        net = tf_util.conv2d(net, 1024, [1, 1],
                             padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training,
                             scope='conv5', bn_decay=self.bn_decay)
        net = tf.squeeze(net, 2)
        net = add_gated_layers(5.5, net, self.is_training)

        # Symmetric function: max pooling
        net = tf.expand_dims(net, 2)
        net = tf_util.max_pool2d(net, [num_point, 1],
                                 padding='VALID', scope='maxpool')

        # MLP on global point cloud vector
        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=self.is_training,
                                      scope='fc1', bn_decay=self.bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=self.is_training,
                              scope='dp1')
        net = tf_util.fully_connected(net, 256, bn=True, is_training=self.is_training,
                                      scope='fc2', bn_decay=self.bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=self.is_training,
                              scope='dp2')
        net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

        return net, end_points


def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
