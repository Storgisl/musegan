import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

class NowbarHybrid:
    def __init__(self, config):
        self.z_intra = tf.keras.Input(shape=(config.z_intra_dim, config.track_dim), dtype=tf.float32, name='z_intra')
        self.z_inter = tf.keras.Input(shape=(config.z_inter_dim,), dtype=tf.float32, name='z_inter')
        self.x = tf.keras.Input(shape=(config.num_bar, config.num_step, config.pitch_range, config.track_dim), dtype=tf.float32, name='x')

class NowbarJamming:
    def __init__(self, config):
        self.z_intra = tf.keras.Input(shape=(config.z_intra_dim, config.track_dim), dtype=tf.float32, name='z_intra')
        self.x = tf.keras.Input(shape=(config.num_bar, config.num_step, config.pitch_range, config.track_dim), dtype=tf.float32, name='x')

class NowbarComposer:
    def __init__(self, config):
        self.z_inter = tf.keras.Input(shape=(config.z_inter_dim,), dtype=tf.float32, name='z_inter')
        self.x = tf.keras.Input(shape=(config.num_bar, config.num_step, config.pitch_range, config.track_dim), dtype=tf.float32, name='x')

class TemporalHybrid:
    def __init__(self, config):
        # Save config
        self.z_intra_dim = config.z_intra_dim
        self.z_inter_dim = config.z_inter_dim
        self.track_dim = config.track_dim
        self.num_phrase = config.num_phrase
        self.num_bar = config.num_bar
        self.num_step = config.num_step
        self.pitch_range = config.pitch_range
        self.batch_size = config.batch_size
        self.z_inter_dim = 32  # or whatever is appropriate
        self.batch_size = 1
        # Define inputs
        self.z_intra_v = tf.compat.v1.placeholder(tf.float32, shape=(None, self.z_intra_dim, self.track_dim), name='z_intra_v')
        self.z_intra_i = tf.compat.v1.placeholder(tf.float32, shape=(None, self.z_intra_dim, self.track_dim), name='z_intra_i')
        self.z_inter_v = tf.compat.v1.placeholder(tf.float32, shape=(None, self.z_inter_dim), name='z_inter_v')
        self.z_inter_i = tf.compat.v1.placeholder(tf.float32, shape=(None, self.z_inter_dim), name='z_inter_i')
        self.x = tf.compat.v1.placeholder(tf.float32, shape=(None, self.num_phrase, self.num_bar, self.num_step, self.pitch_range, self.track_dim), name='x')
        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')

        # Build generator and discriminator
        self.generated = self._build_generator(self.z_inter_v)
        self.fake_x = self.generated  # 👈 Add this line here
        self.prediction = self.generated
        self.prediction_binary = tf.cast(tf.greater(self.generated, 0.5), tf.float32)
        chroma_bins = 12
        reshaped = tf.reshape(self.generated, [-1, self.num_step * self.pitch_range, self.track_dim])
        chroma = []

        for i in range(chroma_bins):
            idxs = list(range(i, self.pitch_range, chroma_bins))
            chroma_i = tf.reduce_sum(tf.gather(reshaped, idxs, axis=1), axis=1)
            chroma.append(chroma_i)

        chroma_tensor = tf.stack(chroma, axis=1)
        self.prediction_chroma = tf.reshape(chroma_tensor, [-1, chroma_bins, self.track_dim])

        self.real_logit, self.real_prob = self._build_discriminator(self.x, reuse=False)
        self.fake_logit, self.fake_prob = self._build_discriminator(self.generated, reuse=True)

        # Losses
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logit, labels=tf.ones_like(self.real_logit)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit, labels=tf.zeros_like(self.fake_logit)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logit, labels=tf.ones_like(self.fake_logit)))

        # Variables
        self.g_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        self.d_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

        # Optimizers
        self.d_optim = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)
        self.z = tf.placeholder(tf.float32, [None, 128], name='z')
        self.z_dim = 128
    def sample_z(self, batch_size):
        return np.random.normal(0, 1, size=(batch_size, self.z_dim))

    def _build_generator(self, z_inter_v):
        with tf.compat.v1.variable_scope('generator'):
            dense = tf.keras.layers.Dense(units=self.num_phrase * self.num_bar * self.num_step * self.pitch_range * self.track_dim)(z_inter_v)
            reshaped = tf.keras.layers.Reshape((self.num_phrase, self.num_bar, self.num_step, self.pitch_range, self.track_dim))(dense)
            return reshaped

    def _build_discriminator(self, x, reuse=False):
        with tf.compat.v1.variable_scope('discriminator', reuse=reuse):
            flat = tf.keras.layers.Flatten()(x)
            dense = tf.keras.layers.Dense(1)(flat)
            logit = dense
            prob = tf.keras.layers.Activation('sigmoid')(logit)
            return logit, prob

    def get_model_info(self, quiet=True):
        if not quiet:
            print("[i] TemporalHybrid: no model info available.")


class TemporalJamming:
    def __init__(self, config):
        self.z_intra_v = tf.keras.Input(shape=(config.z_intra_dim, config.track_dim), dtype=tf.float32, name='z_intra_v')
        self.z_intra_i = tf.keras.Input(shape=(config.z_intra_dim, config.track_dim), dtype=tf.float32, name='z_intra_i')
        self.x = tf.keras.Input(shape=(config.num_phrase, config.num_bar, config.num_step, config.pitch_range, config.track_dim), dtype=tf.float32, name='x')

class TemporalComposer:
    def __init__(self, config):
        self.z_inter_v = tf.keras.Input(shape=(config.z_inter_dim,), dtype=tf.float32, name='z_inter_v')
        self.z_inter_i = tf.keras.Input(shape=(config.z_inter_dim,), dtype=tf.float32, name='z_inter_i')
        self.x = tf.keras.Input(shape=(config.num_phrase, config.num_bar, config.num_step, config.pitch_range, config.track_dim), dtype=tf.float32, name='x')

class RNNComposer:
    def __init__(self, config):
        self.z_inter = tf.keras.Input(shape=(config.z_inter_dim,), dtype=tf.float32, name='z_inter')
        self.x = tf.keras.Input(shape=(config.num_bar, config.num_step, config.pitch_range, config.track_dim), dtype=tf.float32, name='x')

class ImageMNIST:
    def __init__(self, config):
        self.z = tf.keras.Input(shape=(config.z_dim,), dtype=tf.float32, name='z')
        self.x = tf.keras.Input(shape=(28, 28, 1), dtype=tf.float32, name='x')

