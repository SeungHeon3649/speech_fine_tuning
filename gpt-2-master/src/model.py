import numpy as np
import tensorflow as tf
from keras.layers import Layer, Dense

class HParams:
    def __init__(self, n_vocab=0, n_ctx=1024, n_embd=768, n_head=12, n_layer=12):
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer

def default_hparams():
    return HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
    )

def shape_list(x):
    """TensorFlow 2.x에서 동적 모양 다루기."""
    dynamic = tf.shape(x)
    return [dynamic[i] for i in range(len(x.shape))]

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)

def gelu(x):
    return 0.5 * x * (1 + tf.math.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

class LayerNormalization(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight("gamma", shape=input_shape[-1:], initializer="ones", trainable=True)
        self.beta = self.add_weight("beta", shape=input_shape[-1:], initializer="zeros", trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.epsilon) + self.beta

class AttentionMask(Layer):
    def call(self, inputs):
        nd, ns = shape_list(inputs)
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, inputs.dtype)

class MultiHeadAttention(Layer):
    def __init__(self, n_state, hparams, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.n_state = n_state
        self.hparams = hparams

    def build(self, input_shape):
        self.c_attn = Dense(3 * self.n_state, use_bias=False, name="c_attn")
        self.c_proj = Dense(self.n_state, use_bias=False, name="c_proj")
        super(MultiHeadAttention, self).build(input_shape)

    def split_heads(self, x):
        # [batch, sequence, features] -> [batch, heads, sequence, features]
        return tf.transpose(tf.reshape(x, shape_list(x)[:-1] + [self.hparams.n_head, -1]), [0, 2, 1, 3])

    def merge_heads(self, x):
        # Reverse of split_heads
        return tf.reshape(tf.transpose(x, [0, 2, 1, 3]), shape_list(x)[:-2] + [self.n_state])

    def mask_attn_weights(self, w):
        _, _, nd, ns = shape_list(w)
        b = self.add_weight("attention_mask", shape=(1, 1, nd, ns), initializer="zeros", trainable=False)
        m = AttentionMask()(w) * b
        w = w * m - (1 - m) * 1e10
        return w

    def call(self, inputs, past=None):
        c = self.c_attn(inputs)
        q, k, v = tf.split(c, 3, axis=-1)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(v.shape[-1], q.dtype))
        a = self.mask_attn_weights(a)
        a = softmax(a)
        a = tf.matmul(a, v)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a

class PositionWiseFeedForward(Layer):
    def __init__(self, n_state, **kwargs):
        super(PositionWiseFeedForward, self).__init__(**kwargs)
        self.n_state = n_state

    def build(self, input_shape):
        self.c_fc = Dense(self.n_state * 4, activation=gelu, use_bias=True, name="c_fc")
        self.c_proj = Dense(self.n_state, use_bias=True, name="c_proj")
        super(PositionWiseFeedForward, self).build(input_shape)

    def call(self, x):
        h = self.c_fc(x)
        h = self.c_proj(h)
        return h

class Block(Layer):
    def __init__(self, past, hparams, **kwargs):
        super(Block, self).__init__(**kwargs)
        self.past = past
        self.hparams = hparams

    def build(self, input_shape):
        self.ln_1 = LayerNormalization(name="ln_1")
        self.attn = MultiHeadAttention(self.past.shape[-1], self.hparams, name="attn")
        self.ln_2 = LayerNormalization(name="ln_2")
        self.mlp = PositionWiseFeedForward(input_shape[-1], name="mlp")
        super(Block, self).build(input_shape)

    def call(self, x):
        a = self.ln_1(x)
        a = self.attn(a, past=self.past)
        x = x + a
        m = self.ln_2(x)
        m = self.mlp(m)
        x = x + m
        return x

def past_shape(hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

class Model(Layer):
    def __init__(self, hparams, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.hparams = hparams

    def build(self, input_shape):
        self.wpe = self.add_weight("wpe", shape=(self.hparams.n_ctx, self.hparams.n_embd),
                                  initializer="random_normal", trainable=True)
        self.wte = self.add_weight("wte", shape=(self.hparams.n_vocab, self.hparams.n_embd),
                                  initializer="random_normal", trainable=True)
        super(Model, self).build(input_shape)

    def positions_for(self, tokens, past_length):
        batch_size = tf.shape(tokens)[0]
        nsteps = tf.shape(tokens)[1]
        return tf.tile(tf.expand_dims(past_length + tf.range(nsteps), axis=0), [batch_size, 1])

    def call(self, X, past=None, scope='model', reuse=False):
        with tf.name_scope(scope):
            results = {}
            batch, sequence = shape_list(X)

            past_length = 0 if past is None else tf.shape(past)[-2]
            h = tf.gather(self.wte, X) + tf.gather(self.wpe, self.positions_for(X, past_length))

            # Transformer
            presents = []
            pasts = tf.unstack(past, axis=1) if past is not None else [None] * self.hparams.n_layer
            assert len(pasts) == self.hparams.n_layer
            for layer, past in enumerate(pasts):
                h = Block(past, self.hparams, name=f'h{layer}')(h)
                presents.append(past)
            results['present'] = tf.stack(presents, axis=1)
            h = LayerNormalization(name="ln_f")(h)

            # Language model loss
            h_flat = tf.reshape(h, [batch * sequence, self.hparams.n_embd])
            logits = tf.matmul(h_flat, self.wte, transpose_b=True)
            logits = tf.reshape(logits, [batch, sequence, self.hparams.n_vocab])
            results['logits'] = logits
            return results

# Example usage
hparams = HParams(n_vocab=10000, n_ctx=1024, n_embd=768, n_head=12, n_layer=12)
model = Model(hparams)
X = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])  # Replace with your input data
past = None  # Replace with past states if available
results = model(X, past=past)
print(results['logits'])
