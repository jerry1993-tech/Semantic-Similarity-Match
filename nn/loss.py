import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.losses import util as tf_losses_util
from tensorflow.keras.losses import Loss


class LossFunctionWrapper(Loss):
    def __init__(self, fn, reduction=losses_utils.ReductionV2.AUTO, name=None, **kwargs):
        super(LossFunctionWrapper, self).__init__(reduction=reduction, name=name)
        self.fn = fn
        self._fn_kwargs = kwargs

    def call(self, y_true, y_pred):
        if tensor_util.is_tensor(y_pred) and tensor_util.is_tensor(y_true):
            y_pred, y_true = tf_losses_util.squeeze_or_expand_dimensions(y_pred, y_true)
        return self.fn(y_true, y_pred, **self._fn_kwargs)

    def get_config(self):
        config = {}
        for k, v in self._fn_kwargs.items():
            config[k] = K.eval(v) if tf_utils.is_tensor_or_variable(v) else v
        base_config = super(LossFunctionWrapper, self).get_config()
        return {**base_config, **config}


class SparseCategoricalCrossentropyWithMask(LossFunctionWrapper):
    def __init__(self,
                 from_logits=False,
                 reduction=losses_utils.ReductionV2.AUTO,
                 mask_unkid=None,
                 mask_padid=None,
                 name='sparse_categorical_crossentropy_with_mask'):
        super(SparseCategoricalCrossentropyWithMask, self).__init__(sparse_categorical_crossentropy_with_mask,
                                                                    name=name,
                                                                    unkid=mask_unkid,
                                                                    padid=mask_padid,
                                                                    reduction=reduction,
                                                                    from_logits=from_logits)


class CategoricalCrossentropyWithMask(LossFunctionWrapper):
    def __init__(self,
                 from_logits=False,
                 label_smoothing=0,
                 reduction=losses_utils.ReductionV2.AUTO,
                 mask_unkid=None,
                 mask_padid=None,
                 name='categorical_crossentropy_with_mask'):
        super(CategoricalCrossentropyWithMask, self).__init__(categorical_crossentropy_with_mask,
                                                              name=name,
                                                              unkid=mask_unkid,
                                                              padid=mask_padid,
                                                              reduction=reduction,
                                                              from_logits=from_logits,
                                                              label_smoothing=label_smoothing)


def sparse_categorical_crossentropy_with_mask(y_true, y_pred, from_logits=False, axis=-1, unkid=None, padid=None):
    """Computes the sparse categorical crossentropy loss.

    Standalone usage:

    >>> y_true = [1, 2]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss.numpy()
    array([0.0513, 2.303], dtype=float32)

    Args:
        y_true: Ground truth values.
        y_pred: The predicted values.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
        we assume that `y_pred` encodes a probability distribution.
        axis: (Optional) Defaults to -1. The dimension along which the entropy is
        computed.

    Returns:
        Sparse categorical crossentropy loss value.
    """

    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)

    mask = tf.squeeze((y_true != unkid) & (y_true != padid))
    masked_pred = tf.boolean_mask(y_pred, mask)
    masked_true = tf.boolean_mask(y_true, mask)

    return K.sparse_categorical_crossentropy(masked_true, masked_pred)


def categorical_crossentropy_with_mask(y_true, y_pred, from_logits=False, label_smoothing=0, unkid=None, padid=None):
    """Computes the categorical crossentropy loss.

    Standalone usage:

    >>> y_true = [[0, 1, 0], [0, 0, 1]]
    >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
    >>> loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    >>> assert loss.shape == (2,)
    >>> loss.numpy()
    array([0.0513, 2.303], dtype=float32)

    Args:
        y_true: Tensor of one-hot true targets.
        y_pred: Tensor of predicted targets.
        from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
        we assume that `y_pred` encodes a probability distribution.
        label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.

    Returns:
        Categorical crossentropy loss value.
    """

    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    label_smoothing = ops.convert_to_tensor(label_smoothing, dtype=K.floatx())

    def _smooth_labels():
        num_classes = math_ops.cast(array_ops.shape(y_true)[1], y_pred.dtype)
        return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

    y_true = smart_cond.smart_cond(label_smoothing, _smooth_labels, lambda: y_true)

    mask = tf.ones_like(y_true)
    if unkid is not None:
        mask = mask & (y_true[..., unkid] != 1)
    if padid is not None:
        mask = mask & (y_true[..., padid] != 1)
    mask = tf.squeeze(mask)

    masked_pred = tf.boolean_mask(y_pred, mask)
    masked_true = tf.boolean_mask(y_true, mask)

    return K.categorical_crossentropy(masked_true, masked_pred)
