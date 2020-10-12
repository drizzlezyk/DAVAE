import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Dropout, Lambda, Concatenate, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import numpy as np
from tensorflow.python.keras.layers import Layer


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    g = tf.compat.v1.get_default_graph()
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)

    return y


# class GradientReversal(Layer):
#     '''Flip the sign of gradient during training.'''
#     def __init__(self, hp_lambda, **kwargs):
#         super(GradientReversal, self).__init__(**kwargs)
#         self.supports_masking = False
#         self.hp_lambda = hp_lambda
#
#     def build(self, input_shape):
#         self._trainable_weights = []
#
#     def call(self, x, mask=None):
#         return reverse_gradient(x, self.hp_lambda)
#
#     def get_output_shape_for(self, input_shape):
#         return input_shape
#
#     def get_config(self):
#         config = {'hp_lambda': self.hp_lambda}
#         base_config = super(GradientReversal, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

from tensorflow.python.framework import ops

# class FlipGradientBuilder(object):
#     def __init__(self, l=1.0):
#         self.num_calls = 0
#         self.l = l
#
#     def __call__(self, x):
#         grad_name = "FlipGradient%d" % self.num_calls
#         @ops.RegisterGradient(grad_name)
#         def _flip_gradients(op, grad):
#             return [tf.negative(grad) * self.l]
#
#         g = tf.compat.v1.get_default_graph()
#         with g.gradient_override_map({"Identity": grad_name}):
#             y = tf.identity(x)
#
#         self.num_calls += 1
#         return y


@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad


class GradReverse(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)


class DAVAE:
    def __init__(self, input_size, latent_size=16, batches=2, path=''):
        self.input_size = input_size
        self.path = path
        self.dann_vae = None
        self.inputs = None
        self.outputs_x = None
        self.initializers = "glorot_uniform"
        self.optimizer = optimizers.Adam(lr=0.01)
        self.latent_z_size = latent_size
        self.dropout_rate_small = 0.01
        self.dropout_rate_big = 0.1
        self.kernel_regularizer = regularizers.l1_l2(l1=0.00, l2=0.00)
        self.validation_split = 0.0
        self.batches = batches
        self.dropout_rate = 0.01
        callbacks = []
        checkpointer = ModelCheckpoint(filepath=path + "vae_weights.h5", verbose=1, save_best_only=False,
                                       save_weights_only=True)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=10, min_lr=0.0001)
        early_stop = EarlyStopping(monitor='loss', patience=60)
        tensor_board = TensorBoard(log_dir=path + 'logs/')
        callbacks.append(checkpointer)
        callbacks.append(reduce_lr)
        callbacks.append(early_stop)
        callbacks.append(tensor_board)
        self.callbacks = callbacks

    def build(self):
        Relu = "relu"
        inputs_x = Input(shape=(self.input_size,), name='inputs')
        inputs_batch = Input(shape=(self.batches,), name='inputs_batch')
        inputs_loss_weights = Input(shape=(1,), name='inputs_weights')
        inputs = concatenate([inputs_x, inputs_batch])
        hx = Dense(128, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                   name='hx_hidden_layer_x2')(inputs)
        hx = BatchNormalization(center=True, scale=False)(hx)
        hx = Activation(Relu)(hx)
        hx = Dropout(self.dropout_rate_big)(hx)

        hx = Dense(64, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                   name='hx_hidden_layer_x3')(hx)
        hx = BatchNormalization(center=True, scale=False)(hx)
        hx = Activation(Relu)(hx)
        hx = Dropout(self.dropout_rate_small)(hx)

        hx_mean = Dense(self.latent_z_size, kernel_regularizer=self.kernel_regularizer,
                        kernel_initializer=self.initializers,
                        name="hx_mean")(hx)
        hx_log_var = Dense(self.latent_z_size, kernel_regularizer=self.kernel_regularizer,
                           kernel_initializer=self.initializers,
                           name="hx_log_var")(hx)
        hx_z = Lambda(sampling, output_shape=(self.latent_z_size,), name='hx_z')([hx_mean, hx_log_var])
        encoder_hx = Model([inputs_x, inputs_batch], [hx_mean, hx_log_var, hx_z], name='encoder_hx')

        latent_inputs_x = Input(shape=(self.latent_z_size,), name='latent')
        latent_inputs_batch = Input(shape=(self.batches,), name='latent_batch')
        latent_inputs = concatenate([latent_inputs_x, latent_inputs_batch])
        x = Dense(64, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(latent_inputs)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        x = Dropout(self.dropout_rate_big)(x)
        x = Dense(128, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(x)
        x = BatchNormalization(center=True, scale=False)(x)
        x = Activation(Relu)(x)
        x = Dropout(self.dropout_rate_small)(x)
        outputs_x = Dense(self.input_size, kernel_regularizer=self.kernel_regularizer,
                          kernel_initializer=self.initializers, activation="softplus")(x)
        decoder_x = Model([latent_inputs_x, latent_inputs_batch], outputs_x, name='decoder_x')

        latent_inputs_batch = Input(shape=(self.latent_z_size,), name='latent_domain')
        Flip = GradReverse()

        d = Flip(latent_inputs_batch)
        d = Dense(16, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers)(d)
        d = BatchNormalization(center=True, scale=False)(d)
        d = Activation(Relu)(d)
        d = Dropout(0.05)(d)

        d = Dense(self.batches, kernel_regularizer=self.kernel_regularizer, kernel_initializer=self.initializers,
                  activation="softmax")(d)
        domian_classifier = Model(latent_inputs_batch, d, name='domain_classifier')

        outputs_x = decoder_x([encoder_hx([inputs_x, inputs_batch])[2], inputs_batch])
        domain_pred = domian_classifier(encoder_hx([inputs_x, inputs_batch])[2])

        dann_vae = Model([inputs_x, inputs_batch, inputs_loss_weights], [outputs_x, domain_pred], name='vae_mlp')

        inputs_x_value = tf.multiply(inputs_x, inputs_loss_weights)
        outputs_x_value = tf.multiply(outputs_x, inputs_loss_weights)
        reconstruction_loss = mse(inputs_x_value, outputs_x_value)
        # reconstruction_loss = mse(inputs_x, outputs_x)

        noise = tf.math.subtract(inputs_x, outputs_x)
        var = tf.math.reduce_variance(noise)
        reconstruction_loss *= (0.5*self.input_size)/var
        reconstruction_loss += (0.5*self.input_size)/var*tf.math.log(var)

        kl_loss_z = -0.5 * K.sum(1 + hx_log_var - K.square(hx_mean) - K.exp(hx_log_var), axis=-1)

        pred_loss = K.categorical_crossentropy(inputs_batch, domain_pred)*self.input_size*3
        vae_loss = K.mean(reconstruction_loss + kl_loss_z*2 + pred_loss)

        dann_vae.add_loss(vae_loss)
        self.dann_vae = dann_vae
        self.encoder = encoder_hx

    def compile(self):
        self.dann_vae.compile(optimizer=self.optimizer)
        self.dann_vae.summary()

    def train(self, x, batch, loss_weights, batch_size=100, epochs=300, gpu=False):

        history = self.dann_vae.fit({'inputs': x, 'inputs_batch': batch, 'inputs_weights': loss_weights},
                                        epochs=epochs, batch_size=batch_size,
                                        validation_split=self.validation_split, shuffle=True)
        return history

    def embedding(self, x, batches, save=True, use_mean=True):
        [z_mean, z_log_var, z] = self.encoder.predict([x, batches])
        return z_mean


def integration(adata_list, epoch=20, batch_size=256, latent_size=6, sparse=False):
    batch_num = len(adata_list)
    adata = adata_list[0].concatenate(adata_list[1])
    for i in range(2, len(adata_list)):
        adata = adata.concatenate(adata_list[i])
    batch = adata.obs['batch']
    batch = np.array(batch.values, dtype=int)
    loss_weight = batch
    np.minimum(loss_weight, 1)
    print(loss_weight)
    orig_batch = to_categorical(batch)
    if sparse:
        orig_data = adata.X.A
    else:
        orig_data = adata.X
    data, batch, loss_weight = shuffle(orig_data, orig_batch, loss_weight, random_state=0)
    net = DAVAE(input_size=data.shape[1], latent_size=latent_size, batches=batch_num)
    net.build()
    net.compile()
    net.train(data, batch, loss_weight, batch_size=batch_size, epochs=epoch)
    integrate_latent_layer = net.embedding(orig_data, orig_batch)
    adata.obsm['davae'] = integrate_latent_layer
    return adata


