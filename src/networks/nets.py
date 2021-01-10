import layers
import numpy as np
import options
import tensorflow as tf
import util


###################
# Simple CNN      #
###################

def build_cnn_generator(generator_input, img_size, scope='generator', return_mapping=False, reuse=None, training=None, **kwargs):
    if kwargs:
        print(f'Warning! Unused network kwargs ignored: {kwargs}')

    config = options.get_options()
    max_resolution = max(img_size[0], img_size[1])
    initial_lod = int(np.log2(max_resolution)) - int(np.log2(config.initial_resolution)) # level of detail
    initial_res1 = int(img_size[0] / (2 ** initial_lod))
    initial_res2 = int(img_size[1] / (2 ** initial_lod))  # N.B. assumes the resolutions are powers of 2
    res_str = lambda res: f'{initial_res1 * 2 ** res}x{initial_res2 * 2 ** res}'

    num_features_base = 4096  # number of filters (pre-max!) in the first block
    num_features_max = 512

    def num_features(res): # how many filters there are in a certain layer
        return min(int(num_features_base / (2 ** res)), num_features_max)

    x = generator_input
    res = 0

    with tf.variable_scope(scope, reuse=reuse):
        while res <= initial_lod:
            nf = num_features(res)
            with tf.variable_scope(res_str(res)):
                if res == 0:
                    with tf.variable_scope('dense'):
                        initial_num_units = initial_res1 * initial_res2
                        x = layers.dense(x, nf*initial_num_units, gain=np.sqrt(2/initial_num_units), training=training)
                        x = tf.reshape(x, (-1, initial_res1, initial_res2, nf))
                        x = layers.activation(layers.apply_bias(x))
                else:
                    with tf.variable_scope('conv1_up'):
                        x = layers.blur2d(layers.upscale2d_conv2d(x, nf, 3, training=training)) # overgenomen uit StyleGAN
                        x = layers.activation(layers.apply_bias(x))

                with tf.variable_scope('conv2'):
                    x = layers.conv2d(x, nf, 3, training=training)
                    x = layers.activation(layers.apply_bias(x))

                # print('gen')
                # print((x.shape, res_str(res)))

            res += 1

        x = layers.conv2d(x, img_size[-1], 1, gain=1, training=training)
        generator = tf.identity(x, name='generator_output')

    if return_mapping:
        return generator, generator_input
    else:
        return generator


def build_cnn_discriminator(discriminator_input, img_size, return_encoder=False, scope='discriminator', reuse=None, training=None):
    config = options.get_options()
    max_resolution = max(img_size[0], img_size[1])
    initial_lod = int(np.log2(max_resolution)) - int(np.log2(config.initial_resolution))
    initial_res1 = int(img_size[0] / (2 ** initial_lod))
    initial_res2 = int(img_size[1] / (2 ** initial_lod))  # N.B. assumes the resolutions are powers of 2
    res_str = lambda res: f'{initial_res1 * 2 ** res}x{initial_res2 * 2 ** res}'

    num_features_base = 4096  # number of filters (pre-max!) in the first block
    num_features_max = 512

    def num_features(res):
        return min(int(num_features_base / (2 ** res)), num_features_max)

    x = discriminator_input
    res = initial_lod

    with tf.variable_scope(scope, reuse=reuse):
        while res > 0:
            nf = num_features(res)
            with tf.variable_scope(res_str(res)):
                with tf.variable_scope('conv1'):
                    x = layers.conv2d(x, nf, 3, training=training)
                    x = layers.activation(layers.apply_bias(x))
                with tf.variable_scope('conv2_down'):
                    x = layers.conv2d_downscale2d(layers.blur2d(x), nf, 3, training=training)
                    x = layers.activation(layers.apply_bias(x))

            # print('disc')
            # print((x.shape, res_str(res)))
            res -= 1

        x = tf.identity(x, name='encoder_output')
        enc_out = layers.flatten(x)
        nf = num_features(res)

        if config.training_method != 'gan':
            with tf.variable_scope('bottleneck'):
                enc_out = layers.dense(x, config.autoencoder_latent_dim, gain=1, training=training)
                enc_out = layers.activation(layers.apply_bias(enc_out))

        # Gan part (not used in autoencoder)
        # with tf.variable_scope('minibatch_sim'):
        #     x, num_params = layers.minibatch_similarity(x, training=training)
        # with tf.variable_scope('conv'):
        #     x = layers.conv2d(x, nf, 3, training=training)
        #     x = layers.activation(layers.apply_bias(x))
        # with tf.variable_scope('dense1'):
        #     x = layers.dense(x, nf, training=training)
        #     x = layers.activation(layers.apply_bias(x))
        # with tf.variable_scope('dense2'):
        #     x = layers.dense(x, 1, gain=1, training=training)
        #     x = layers.apply_bias(x)

        # discriminator = tf.identity(x, name='discriminator_output')

    if return_encoder:
        return discriminator, enc_out
    else:
        return discriminator

