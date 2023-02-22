import numpy as np
import os
import sys

local_dir = os.path.abspath('..')
sys.path.append(local_dir)
from nn.activations import get_activation
import tensorflow as tf
import json


class PointWiseFeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dff_act="relu", name="ffn", **kwargs):
        super(PointWiseFeedForward, self).__init__(name=name, **kwargs)
        self.dff = dff
        self.d_model = d_model
        self.dff_act = dff_act

    def build(self, input_shape):
        self.intermediate = tf.keras.layers.Dense(self.dff, activation=self.dff_act)  # (batch_size, seq_len, dff)
        self.out = tf.keras.layers.Dense(self.d_model)  # (batch_size, seq_len, d_model)

        self.intermediate.build(input_shape)
        self.out.build(input_shape[:-2] + [self.dff])

        self.built = True

    def call(self, inputs, **kwargs):
        out = self.intermediate(inputs)
        out = self.out(out)
        return out


class ScaledDotProductAttention(tf.keras.layers.Layer):
    def __init__(self, attention_dropout_rate=0.1, name="scaled_dot_product_attention", **kwargs):
        super(ScaledDotProductAttention, self).__init__(name=name, **kwargs)
        self.dropout = tf.keras.layers.Dropout(attention_dropout_rate)

    def call(self, inputs, mask=None):
        """Calculate the attention weights.
            q, k, v must have matching leading dimensions.
            k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
            The mask has different shapes depending on its type(padding or look ahead)
            but it must be broadcastable for addition.

            Args:
              inputs: [q, k, v]
                q: query shape == (..., seq_len_q, depth)
                k: key shape == (..., seq_len_k, depth)
                v: value shape == (..., seq_len_v, depth_v)
              mask: Float tensor with shape broadcastable
                    to (..., seq_len_q, seq_len_k). Defaults to None.

            Returns:
              output, attention_weights
            """
        q, k, v = inputs
        matmul_qk = tf.matmul(q, k,
                              transpose_b=True)  # (N, heads, seq_q, d_k)*(N, heads, d_k, seq_k)=(N, heads, seq_q, seq_k)

        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        # -inf经过softmax后会接近于0，一方面保证句子里pad的部分几乎不会分到注意力，另一方面也保证解码的时候当前时间步后面的部分句子不会分配到注意力
        if mask is not None:
            scaled_attention_logits += (mask * -10000.0)

            # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
        attention_weights = tf.keras.layers.Softmax()(scaled_attention_logits)  # (..., seq_len_q, seq_len_k)

        # FIXME: 可能不需要dropout https://github.com/kaituoxu/Speech-Transformer/blob/master/src/transformer/attention.py#L83
        attention_weights = self.dropout(attention_weights)

        output = tf.matmul(attention_weights, v)  # (..., seq_len_v, depth) 实际上是(..., seq_len_q, depth)，只是三种len都一样

        return output


class EmbeddingUnique(tf.keras.layers.Embedding):
    def call(self, inputs, use_one_hot=False):
        if inputs.dtype != 'int32' and inputs.dtype != 'int64':
            inputs = tf.cast(inputs, 'int32')

        ids = tf.convert_to_tensor(inputs)
        shape = tf.shape(ids)
        ids_flat = tf.reshape(ids, [-1])
        unique_ids, idx = tf.unique(ids_flat)
        if use_one_hot:
            one_hot_data = tf.one_hot(unique_ids, depth=self.input_dim, dtype=self._dtype)
            unique_embeddings = tf.matmul(one_hot_data, self.embeddings)
        else:
            unique_embeddings = tf.nn.embedding_lookup(self.embeddings, unique_ids)
        embeds_flat = tf.gather(unique_embeddings, idx)
        embed_shape = tf.concat([shape, [self.output_dim]], 0)
        embeddings = tf.reshape(embeds_flat, embed_shape)

        return embeddings


class PositionEmbedding(tf.keras.layers.Embedding):
    def call(self, inputs):
        if inputs.dtype != 'int32' and inputs.dtype != 'int64':
            inputs = tf.cast(inputs, 'int32')
        seq_len = inputs
        embeds = tf.slice(self.embeddings, [0, 0], [seq_len, -1])
        return embeds


class TwoDAttention(tf.keras.layers.Layer):
    def __init__(self, n=64, c=64, attention_dropout_rate=0.1, name="2d_attention", **kwargs):
        super(TwoDAttention, self).__init__(name=name, **kwargs)

        self.n = n
        self.c = c
        self.attention_dropout_rate = attention_dropout_rate

    def build(self, input_shape):
        self.convq = tf.keras.layers.Conv2D(self.c, 3, 1, 'same', kernel_initializer='glorot_normal')
        self.convk = tf.keras.layers.Conv2D(self.c, 3, 1, 'same', kernel_initializer='glorot_normal')
        self.convv = tf.keras.layers.Conv2D(self.c, 3, 1, 'same', kernel_initializer='glorot_normal')
        self.conv = tf.keras.layers.Conv2D(self.n, 3, 1, 'same', kernel_initializer='glorot_normal')
        self.bnq = tf.keras.layers.BatchNormalization()
        self.bnk = tf.keras.layers.BatchNormalization()
        self.bnv = tf.keras.layers.BatchNormalization()

        self.scaled_dot_product_attention = ScaledDotProductAttention(
            attention_dropout_rate=self.attention_dropout_rate)
        self.ln = tf.keras.layers.LayerNormalization()

        self.final_conv1 = tf.keras.layers.Conv2D(self.n, 3, 1, 'same', activation='relu',
                                                  kernel_initializer='glorot_normal')
        self.final_conv2 = tf.keras.layers.Conv2D(self.n, 3, 1, 'same', kernel_initializer='glorot_normal')
        self.bnf1 = tf.keras.layers.BatchNormalization()
        self.bnf2 = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')

        # fused LayerNormalization tf-lite 暂不支持
        self.ln.build(input_shape)
        self.ln._fused = False

        self.built = True

    def call(self, inputs, training=1):
        """
        :param training:
        :param inputs: B*T*D*n
        :return: B*T*D*n
        """
        residual = inputs
        # batch_size = tf.shape(inputs)[0]
        q = self.bnq(self.convq(inputs), training=training)
        k = self.bnk(self.convk(inputs), training=training)
        v = self.bnv(self.convv(inputs), training=training)

        q_time = tf.transpose(q, [0, 3, 1, 2])
        k_time = tf.transpose(k, [0, 3, 1, 2])
        v_time = tf.transpose(v, [0, 3, 1, 2])

        q_fre = tf.transpose(q, [0, 3, 2, 1])
        k_fre = tf.transpose(k, [0, 3, 2, 1])
        v_fre = tf.transpose(v, [0, 3, 2, 1])

        scaled_attention_time = self.scaled_dot_product_attention([q_time, k_time, v_time], None)  # B*c*T*D
        scaled_attention_fre = self.scaled_dot_product_attention([q_fre, k_fre, v_fre], None)  # B*c*D*T

        scaled_attention_time = tf.transpose(scaled_attention_time, [0, 2, 3, 1])
        scaled_attention_fre = tf.transpose(scaled_attention_fre, [0, 3, 2, 1])

        out = tf.concat([scaled_attention_time, scaled_attention_fre], -1)  # B*T*D*2c

        out = self.ln(self.conv(out) + residual)  # B*T*D*n

        final_out = self.bnf1(self.final_conv1(out), training=training)
        final_out = self.bnf2(self.final_conv2(final_out), training=training)

        final_out = self.act(final_out + out)

        return final_out


class DownSample(tf.keras.layers.Layer):
    def __init__(self, n=64, times=4, k_size=3, stride=(2, 1), name="down_sample", **kwargs):
        super(DownSample, self).__init__(name=name, **kwargs)
        self.n = n
        self.times = times
        self.k_size = k_size
        self.stride = stride

    def build(self, input_shape):
        self.downsampling = [
            tf.keras.layers.Conv2D(self.n, self.k_size, self.stride, 'same', activation='tanh',
                                   kernel_initializer='glorot_normal') for _ in range(self.times)
        ]

        self.bns = [tf.keras.layers.BatchNormalization() for _ in range(self.times)]

        self.built = True

    def call(self, inputs, training=1):
        """
        :param training:
        :param inputs: B*T*D*n
        :return: B*T*D*c
        """

        out = tf.cast(inputs, tf.float32)

        for index in range(self.times):
            out = self.bns[index](self.downsampling[index](out), training=training)
        return out


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, pe_max_len, d_model, name="positional_encoding", **kwargs):
        super(PositionalEncoding, self).__init__(name=name, **kwargs)
        self.pos_encoding = self.positional_encoding(pe_max_len, d_model)

    def call(self, inputs, **kwargs):
        out = inputs + self.pos_encoding[:, :inputs.shape[1], :]
        return out

    @staticmethod
    def positional_encoding(position, d_model):
        def get_angles(pos, i, d_model):
            angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
            return pos * angle_rates

        angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)  # (1,pe_max_len, d_model)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, attention_dropout_rate=0.1, name="multi_head_attention", **kwargs):
        super(MultiHeadAttention, self).__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.attention_dropout_rate = attention_dropout_rate

    def build(self, input_shape):
        self.wq = tf.keras.layers.Dense(self.d_model)  # (feature_in_dim, d_model) 第一维不是batch size因为可以broadcast
        self.wk = tf.keras.layers.Dense(self.d_model)  # (feature_in_dim, d_model)
        self.wv = tf.keras.layers.Dense(self.d_model)  # (feature_in_dim, d_model)
        self.scaled_dot_product_attention = ScaledDotProductAttention(
            attention_dropout_rate=self.attention_dropout_rate)
        self.dense = tf.keras.layers.Dense(self.d_model)  # (feature_in_dim, d_model)

        self.wq.build(input_shape[0])  # (batch_size, seq_len, d_model)
        self.wk.build(input_shape[1])
        self.wv.build(input_shape[2])
        self.dense.build(input_shape[2])

        self.built = True

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))  # -1 for seq_len,
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None):
        """
        :param inputs: [v, q, k]
                v: input data , shape(batch_size, seq_len, feature_in_dim)
                k: input data , shape(batch_size, seq_len, feature_in_dim)
                q: input data , shape(batch_size, seq_len, feature_in_dim)
        :param mask: padding mask, shape(batch_size, 1, 1, seq_len) # 1 for broadcast
        :return:
        """

        v, k, q = inputs[0], inputs[1], inputs[2]
        # batch_size = q.shape[0]

        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention = self.scaled_dot_product_attention([q, k, v], mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_v, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_v, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_v, d_model)

        return output


class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dff_act="relu", dropout_rate=0.1, attention_dropout_rate=0.1,
                 name="TransformerEncoderLayer", **kwargs):
        super(TransformerEncoderLayer, self).__init__(name=name, **kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.attention_dropout_rate = attention_dropout_rate
        self.dff = dff
        self.dff_act = dff_act
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.mha = MultiHeadAttention(self.d_model, self.num_heads, attention_dropout_rate=self.attention_dropout_rate)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-12)

        self.ffn = PointWiseFeedForward(self.d_model, self.dff, self.dff_act)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-12)

        self.mha.build([input_shape, input_shape, input_shape])
        self.layernorm1.build(input_shape)
        self.ffn.build(input_shape)
        self.layernorm2.build(input_shape)
        self.built = True

    def call(self, inputs, mask=None):
        attn_output = self.mha([inputs, inputs, inputs], mask=mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class SelfAttentionMask(tf.keras.layers.Layer):
    def call(self, inputs, **kwargs):
        mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
        # 添加额外的维度来将填充加到注意力对数（logits）
        return mask[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


class AlbertEncoder(tf.keras.layers.Layer):
    """ALBERT (https://arxiv.org/abs/1810.04805) text encoder network.

      This network implements the encoder described in the paper "ALBERT: A Lite
      BERT for Self-supervised Learning of Language Representations"
      (https://arxiv.org/abs/1909.11942).

      Compared with BERT (https://arxiv.org/abs/1810.04805), ALBERT refactorizes
      embedding parameters into two smaller matrices and shares parameters
      across layers.

      The default values for this object are taken from the ALBERT-Base
      implementation described in the paper.

      Arguments:
        vocab_size: The size of the token vocabulary.
        embedding_width: The width of the word embeddings. If the embedding width
          is not equal to hidden size, embedding parameters will be factorized into
          two matrices in the shape of ['vocab_size', 'embedding_width'] and
          ['embedding_width', 'hidden_size'] ('embedding_width' is usually much
          smaller than 'hidden_size').
        hidden_size: The size of the transformer hidden layers.
        num_hidden_layers: The number of transformer layers.
        num_attention_heads: The number of attention heads for each transformer. The
          hidden size must be divisible by the number of attention heads.
        max_position_embeddings: The maximum sequence length that this encoder can
          consume. If None, max_sequence_length uses the value from sequence length.
          This determines the variable shape for positional embeddings.
        type_vocab_size: The number of types that the 'type_ids' input can take.
        intermediate_size: The intermediate size for the transformer layers.
        activation: The activation to use for the transformer layers.
        dropout_rate: The dropout rate to use for the transformer layers.
        attention_dropout_rate: The dropout rate to use for the attention layers
          within the transformer layers.
        initializer: The initialzer to use for all weights in this encoder.
      """

    def __init__(self,
                 vocab_size,
                 embedding_width=128,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 num_hidden_groups=1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 intermediate_size=3072,
                 activation="gelu",
                 dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 initializer_range=0.02,
                 name="albert_transformer_encoder_layer",
                 **kwargs):
        super(AlbertEncoder, self).__init__(name=name, **kwargs)

        self._initializer = tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
        self._activation = get_activation(activation)

        self.embedding_width = embedding_width
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.dropout_rate = dropout_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_groups = num_hidden_groups
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_dropout_rate = attention_dropout_rate

    def build(self, input_shape):

        self.embedding = EmbeddingUnique(input_dim=self.vocab_size, output_dim=self.embedding_width,
                                         name='word_embeddings')

        # Always uses dynamic slicing for simplicity.
        self.position_embedding = PositionEmbedding(input_dim=self.max_position_embeddings,
                                                    output_dim=self.embedding_width, name='position_embeddings')

        self.type_embedding = EmbeddingUnique(input_dim=self.type_vocab_size, output_dim=self.embedding_width,
                                              name='type_embeddings')

        self.embedding_normalization = tf.keras.layers.LayerNormalization(name='embeddings/layer_norm', epsilon=1e-12,
                                                                          dtype=tf.float32)

        self.embedding_dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

        self.embedding_projection = tf.keras.layers.Dense(self.hidden_size, name='embedding_projection')

        self.self_attention_mask = SelfAttentionMask()

        # transformer layer
        self.transformer_layers = []
        last_name = None
        for layer_idx in range(self.num_hidden_layers):
            group_idx = int(layer_idx / self.num_hidden_layers * self.num_hidden_groups)
            if group_idx == last_name:
                layer = self.transformer_layers[-1]
            else:
                layer = TransformerEncoderLayer(d_model=self.hidden_size,
                                                num_heads=self.num_attention_heads,
                                                dff=self.intermediate_size,
                                                dff_act=self._activation,
                                                dropout_rate=self.dropout_rate,
                                                attention_dropout_rate=self.attention_dropout_rate,
                                                name='transformer/layer_%d' % group_idx)
            last_name = group_idx
            self.transformer_layers.append(layer)

        self.cls_output_layer = tf.keras.layers.Dense(units=self.hidden_size, activation='tanh',
                                                      name='pooler_transform')

        self.embedding.build(input_shape[0])
        self.position_embedding.build(input_shape[0][1])
        self.type_embedding.build(input_shape[1])
        self.embedding_normalization.build(input_shape[0] + [self.embedding_width])
        self.embedding_projection.build(input_shape[0] + [self.embedding_width])
        for layer in self.transformer_layers:
            layer.build(input_shape[0] + [self.hidden_size])
        self.cls_output_layer.build(input_shape[0] + [self.hidden_size])

        self.built = True

    def call(self, inputs, **kwargs):
        word_ids = inputs[0]
        type_ids = inputs[1]

        word_embeddings = self.embedding(word_ids)
        position_embeddings = self.position_embedding(word_embeddings.shape[1])
        type_embeddings = self.type_embedding(type_ids, use_one_hot=False)
        embeddings = word_embeddings + position_embeddings + type_embeddings
        embeddings = self.embedding_normalization(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        # We project the 'embedding' output to 'hidden_size' if it is not already 'hidden_size'.
        if self.embedding_width != self.hidden_size:
            embeddings = self.embedding_projection(embeddings)

        data = embeddings
        attention_mask = self.self_attention_mask(word_ids)
        for layer in self.transformer_layers:
            data = layer(data, mask=attention_mask)

        first_token_tensor = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(data)
        cls_output = self.cls_output_layer(first_token_tensor)

        outputs = [data, cls_output]

        return outputs

    def get_layer(self, name):
        if name == "word_embeddings":
            return self.embedding
        elif name == "position_embeddings":
            return self.position_embedding
        elif name == "type_embeddings":
            return self.type_embedding
        elif name == "embeddings/layer_norm":
            return self.embedding_normalization
        elif name == "embedding_projection":
            return self.embedding_projection
        elif name.startswith("transformer/layer_"):
            return self.transformer_layers[int(name.split("_")[1])]
        elif name == "pooler_transform":
            return self.cls_output_layer

    @staticmethod
    def get_albert_encoder(bert_config, name="albert"):
        """get transformer encoder model
        """

        kwargs = dict(vocab_size=bert_config.vocab_size,
                      hidden_size=bert_config.hidden_size,
                      num_hidden_layers=bert_config.num_hidden_layers,
                      num_attention_heads=bert_config.num_attention_heads,
                      intermediate_size=bert_config.intermediate_size,
                      activation=bert_config.hidden_act,
                      dropout_rate=bert_config.hidden_dropout_prob,
                      attention_dropout_rate=bert_config.attention_probs_dropout_prob,
                      max_position_embeddings=bert_config.max_position_embeddings,
                      type_vocab_size=bert_config.type_vocab_size,
                      initializer_range=bert_config.initializer_range,
                      embedding_width=bert_config.embedding_size,
                      num_hidden_groups=bert_config.num_hidden_groups,
                      name=name)

        return AlbertEncoder(**kwargs)

    def load_albert(self, bert_config, init_checkpoint, config):
        """
            the checkpoint file should be the same with https://github.com/google-research/albert
            :param bert_config: bert config json
            :param init_checkpoint: ckpt path, like path/to/bert_model.ckpt
            """
        variables = tf.train.load_checkpoint(init_checkpoint)
        mapping_dict = self.mapping_variable_name(config)
        # embedding weights
        self.get_layer("word_embeddings").set_weights(
            [variables.get_tensor(mapping_dict["bert/embeddings/word_embeddings"])])
        self.get_layer("position_embeddings").set_weights(
            [variables.get_tensor(mapping_dict["bert/embeddings/position_embeddings"])])
        self.get_layer("type_embeddings").set_weights(
            [variables.get_tensor(mapping_dict["bert/embeddings/token_type_embeddings"])])

        self.get_layer("embeddings/layer_norm").set_weights(
            [variables.get_tensor(mapping_dict["bert/embeddings/LayerNorm/gamma"]),
             variables.get_tensor(mapping_dict["bert/embeddings/LayerNorm/beta"])])

        if bert_config.albert_type == "google":
            self.get_layer("embedding_projection").set_weights([
                variables.get_tensor(mapping_dict["bert/encoder/embedding_hidden_mapping_in/kernel"]),
                variables.get_tensor(mapping_dict["bert/encoder/embedding_hidden_mapping_in/bias"])
            ])

        # multi attention weights
        for i in range(bert_config.num_hidden_layers):
            self.get_layer("transformer/layer_{}".format(i)).set_weights([
                variables.get_tensor(
                    mapping_dict["bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/kernel"]),
                variables.get_tensor(
                    mapping_dict["bert/encoder/transformer/group_0/inner_group_0/attention_1/self/query/bias"]),
                variables.get_tensor(
                    mapping_dict["bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/kernel"]),
                variables.get_tensor(
                    mapping_dict["bert/encoder/transformer/group_0/inner_group_0/attention_1/self/key/bias"]),
                variables.get_tensor(
                    mapping_dict["bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/kernel"]),
                variables.get_tensor(
                    mapping_dict["bert/encoder/transformer/group_0/inner_group_0/attention_1/self/value/bias"]),
                variables.get_tensor(
                    mapping_dict["bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/kernel"]),
                variables.get_tensor(
                    mapping_dict["bert/encoder/transformer/group_0/inner_group_0/attention_1/output/dense/bias"]),
                variables.get_tensor(mapping_dict["bert/encoder/transformer/group_0/inner_group_0/LayerNorm/gamma"]),
                variables.get_tensor(mapping_dict["bert/encoder/transformer/group_0/inner_group_0/LayerNorm/beta"]),
                variables.get_tensor(
                    mapping_dict["bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/kernel"]),
                variables.get_tensor(
                    mapping_dict["bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/dense/bias"]),
                variables.get_tensor(mapping_dict[
                                         "bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/kernel"]),
                variables.get_tensor(mapping_dict[
                                         "bert/encoder/transformer/group_0/inner_group_0/ffn_1/intermediate/output/dense/bias"]),
                variables.get_tensor(mapping_dict["bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/gamma"]),
                variables.get_tensor(mapping_dict["bert/encoder/transformer/group_0/inner_group_0/LayerNorm_1/beta"]),
            ])

        self.get_layer("pooler_transform").set_weights([
            variables.get_tensor(mapping_dict["bert/pooler/dense/kernel"]),
            variables.get_tensor(mapping_dict["bert/pooler/dense/bias"]),
        ])

        init_vars = tf.train.list_variables(init_checkpoint)
        for name, shape in init_vars:
            if name.startswith("bert"):
                print(f"{name}, shape={shape}, *INIT FROM CKPT SUCCESS*")

    def mapping_variable_name(self, config):
        with open(config.variable_mapping_file) as mf:
            name_mapping = json.load(mf)
        if config.use_slimmed_albert:
            return name_mapping
        else:
            return dict([(k, k) for k, v in name_mapping.items()])


class BertEncoder(tf.keras.layers.Layer):
    """BERT text encoder network.

      This network implements the encoder described in the paper "BERT:
      Pre-training of Deep Bidirectional Transformers for Language
      Understanding" (https://arxiv.org/abs/1810.04805).

      Arguments:
        vocab_size: The size of the token vocabulary.
        embedding_width: The width of the word embeddings. If the embedding width
          is not equal to hidden size, embedding parameters will be factorized into
          two matrices in the shape of ['vocab_size', 'embedding_width'] and
          ['embedding_width', 'hidden_size'] ('embedding_width' is usually much
          smaller than 'hidden_size').
        hidden_size: The size of the transformer hidden layers.
        num_hidden_layers: The number of transformer layers.
        num_attention_heads: The number of attention heads for each transformer. The
          hidden size must be divisible by the number of attention heads.
        max_position_embeddings: The maximum sequence length that this encoder can
          consume. If None, max_sequence_length uses the value from sequence length.
          This determines the variable shape for positional embeddings.
        type_vocab_size: The number of types that the 'type_ids' input can take.
        intermediate_size: The intermediate size for the transformer layers.
        activation: The activation to use for the transformer layers.
        dropout_rate: The dropout rate to use for the transformer layers.
        attention_dropout_rate: The dropout rate to use for the attention layers
          within the transformer layers.
        initializer: The initialzer to use for all weights in this encoder.
      """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 intermediate_size=3072,
                 activation="gelu",
                 dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 initializer_range=0.02,
                 name="bert_transformer_encoder_layer",
                 **kwargs):
        super(BertEncoder, self).__init__(name=name, **kwargs)

        self._initializer = tf.keras.initializers.TruncatedNormal(stddev=initializer_range)
        self._activation = get_activation(activation)

        self.embedding_width = hidden_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.dropout_rate = dropout_rate
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.attention_dropout_rate = attention_dropout_rate

    def build(self, input_shape):

        self.embedding = EmbeddingUnique(input_dim=self.vocab_size, output_dim=self.embedding_width,
                                         name='word_embeddings')

        self.position_embedding = PositionEmbedding(input_dim=self.max_position_embeddings,
                                                    output_dim=self.embedding_width, name='position_embeddings')

        self.type_embedding = EmbeddingUnique(input_dim=self.type_vocab_size, output_dim=self.embedding_width,
                                              name='type_embeddings')

        self.embedding_normalization = tf.keras.layers.LayerNormalization(name='embeddings/layer_norm', epsilon=1e-12,
                                                                          dtype=tf.float32)

        self.embedding_dropout = tf.keras.layers.Dropout(rate=self.dropout_rate)

        self.self_attention_mask = SelfAttentionMask()

        # transformer layer
        self.transformer_layers = []
        for i in range(self.num_hidden_layers):
            layer = TransformerEncoderLayer(d_model=self.hidden_size,
                                            num_heads=self.num_attention_heads,
                                            dff=self.intermediate_size,
                                            dff_act=self._activation,
                                            dropout_rate=self.dropout_rate,
                                            attention_dropout_rate=self.attention_dropout_rate,
                                            name='transformer/layer_%d' % i)
            self.transformer_layers.append(layer)

        self.cls_output_layer = tf.keras.layers.Dense(units=self.hidden_size, activation='tanh',
                                                      name='pooler_transform')

        self.embedding.build(input_shape[0])
        self.position_embedding.build(input_shape[0][1])
        self.type_embedding.build(input_shape[1])
        self.embedding_normalization.build(input_shape[0] + [self.embedding_width])
        for layer in self.transformer_layers:
            layer.build(input_shape[0] + [self.hidden_size])
        self.cls_output_layer.build(input_shape[0] + [self.hidden_size])

        self.built = True

    def call(self, inputs, **kwargs):
        word_ids = inputs[0]
        type_ids = inputs[1]

        word_embeddings = self.embedding(word_ids)
        position_embeddings = self.position_embedding(word_embeddings.shape[1])
        type_embeddings = self.type_embedding(type_ids, use_one_hot=False)
        embeddings = word_embeddings + position_embeddings + type_embeddings
        embeddings = self.embedding_normalization(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        data = embeddings
        attention_mask = self.self_attention_mask(word_ids)
        all_layer_output = []
        all_layer_cls = []
        for layer in self.transformer_layers:
            data = layer(data, mask=attention_mask)
            all_layer_output.append(data)
            # all_layer_cls.append(tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(data))

        cls_output = tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(data)
        # cls_output = self.cls_output_layer(first_token_tensor)

        outputs = [data, cls_output, all_layer_output]

        return outputs

    def get_layer(self, name):
        if name == "word_embeddings":
            return self.embedding
        elif name == "position_embeddings":
            return self.position_embedding
        elif name == "type_embeddings":
            return self.type_embedding
        elif name == "embeddings/layer_norm":
            return self.embedding_normalization
        elif name.startswith("transformer/layer_"):
            return self.transformer_layers[int(name.split("_")[1])]
        elif name == "pooler_transform":
            return self.cls_output_layer

    @staticmethod
    def get_bert_encoder(bert_config, name="bert"):
        """get transformer encoder model
        """

        kwargs = dict(vocab_size=bert_config.vocab_size,
                      hidden_size=bert_config.hidden_size,
                      num_hidden_layers=bert_config.num_hidden_layers,
                      num_attention_heads=bert_config.num_attention_heads,
                      intermediate_size=bert_config.intermediate_size,
                      activation=bert_config.hidden_act,
                      dropout_rate=bert_config.hidden_dropout_prob,
                      attention_dropout_rate=bert_config.attention_probs_dropout_prob,
                      max_position_embeddings=bert_config.max_position_embeddings,
                      type_vocab_size=bert_config.type_vocab_size,
                      initializer_range=bert_config.initializer_range,
                      name=name)

        return BertEncoder(**kwargs)

    def load_bert(self, bert_config, init_checkpoint):
        """
            the checkpoint file should be the same with https://github.com/google-research/bert
            :param bert_config: bert config json
            :param init_checkpoint: ckpt path, like path/to/bert_model.ckpt
            """
        variables = tf.train.load_checkpoint(init_checkpoint)
        # print(json.dumps(variables.get_variable_to_shape_map(), indent=4))

        # embedding weights
        self.get_layer("word_embeddings").set_weights([variables.get_tensor("bert/embeddings/word_embeddings")])
        self.get_layer("position_embeddings").set_weights([variables.get_tensor("bert/embeddings/position_embeddings")])
        self.get_layer("type_embeddings").set_weights([variables.get_tensor("bert/embeddings/token_type_embeddings")])

        self.get_layer("embeddings/layer_norm").set_weights(
            [variables.get_tensor("bert/embeddings/LayerNorm/gamma"),
             variables.get_tensor("bert/embeddings/LayerNorm/beta")])

        for i in range(bert_config.num_hidden_layers):
            self.get_layer("transformer/layer_{}".format(i)).set_weights([
                variables.get_tensor("bert/encoder/layer_{}/attention/self/query/kernel".format(i)),
                variables.get_tensor("bert/encoder/layer_{}/attention/self/query/bias".format(i)),
                variables.get_tensor("bert/encoder/layer_{}/attention/self/key/kernel".format(i)),
                variables.get_tensor("bert/encoder/layer_{}/attention/self/key/bias".format(i)),
                variables.get_tensor("bert/encoder/layer_{}/attention/self/value/kernel".format(i)),
                variables.get_tensor("bert/encoder/layer_{}/attention/self/value/bias".format(i)),
                variables.get_tensor("bert/encoder/layer_{}/attention/output/dense/kernel".format(i)),
                variables.get_tensor("bert/encoder/layer_{}/attention/output/dense/bias".format(i)),
                variables.get_tensor("bert/encoder/layer_{}/attention/output/LayerNorm/gamma".format(i)),
                variables.get_tensor("bert/encoder/layer_{}/attention/output/LayerNorm/beta".format(i)),
                variables.get_tensor("bert/encoder/layer_{}/intermediate/dense/kernel".format(i)),
                variables.get_tensor("bert/encoder/layer_{}/intermediate/dense/bias".format(i)),
                variables.get_tensor("bert/encoder/layer_{}/output/dense/kernel".format(i)),
                variables.get_tensor("bert/encoder/layer_{}/output/dense/bias".format(i)),
                variables.get_tensor("bert/encoder/layer_{}/output/LayerNorm/gamma".format(i)),
                variables.get_tensor("bert/encoder/layer_{}/output/LayerNorm/beta".format(i)),
            ])

        self.get_layer("pooler_transform").set_weights([
            variables.get_tensor("bert/pooler/dense/kernel"),
            variables.get_tensor("bert/pooler/dense/bias"),
        ])

        init_vars = tf.train.list_variables(init_checkpoint)
        for name, shape in init_vars:
            if name.startswith("bert"):
                print(f"{name}, shape={shape}, *INIT FROM CKPT SUCCESS*")