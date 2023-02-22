import tensorflow as tf
import tensorflow_addons as tfa


class F1ScoreWithMask(tfa.metrics.F1Score):
    def __init__(self, mask_unkid=None, mask_padid=None, **kwargs):
        super().__init__(**kwargs)
        self.unkid = mask_unkid
        self.padid = mask_padid

    def update_state(self, y_true, y_pred, sample_weight=None):

        mask = tf.ones_like(y_true)
        if self.unkid is not None:
            mask = mask & (y_true[..., self.unkid] != 1)
        if self.padid is not None:
            mask = mask & (y_true[..., self.padid] != 1)
        mask = tf.squeeze(mask)

        y_pred = tf.boolean_mask(y_pred, mask)
        y_true = tf.boolean_mask(y_true, mask)

        if self.threshold is None:
            threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-12)
        else:
            y_pred = y_pred > self.threshold

        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)

        def _weighted_sum(val, sample_weight):
            if sample_weight is not None:
                val = tf.math.multiply(val, tf.expand_dims(sample_weight, 1))
            return tf.reduce_sum(val, axis=self.axis)

        self.true_positives.assign_add(_weighted_sum(y_pred * y_true, sample_weight))
        self.false_positives.assign_add(_weighted_sum(y_pred * (1 - y_true), sample_weight))
        self.false_negatives.assign_add(_weighted_sum((1 - y_pred) * y_true, sample_weight))
        self.weights_intermediate.assign_add(_weighted_sum(y_true, sample_weight))


class SparseF1ScoreWithMask(tfa.metrics.F1Score):
    # `use_crf_mask` is valid when `use_crf=True`
    def __init__(self, mask_unkid=None, mask_padid=None, use_crf=False, **kwargs):
        super().__init__(**kwargs)
        self.unkid = mask_unkid
        self.padid = mask_padid
        self.use_crf = use_crf

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.use_crf:
            mask = tf.squeeze((y_true != self.unkid) & (y_true != self.padid))
            y_pred = tf.boolean_mask(y_pred, mask)
            y_true = tf.boolean_mask(y_true, mask)
            y_pred = tf.squeeze(tf.one_hot(y_pred, depth=self.num_classes))
            y_true = tf.squeeze(tf.one_hot(y_true, depth=self.num_classes))
        else:
            mask = tf.squeeze((y_true != self.unkid) & (y_true != self.padid))
            y_pred = tf.boolean_mask(y_pred, mask)
            y_true = tf.boolean_mask(y_true, mask)

            y_true = tf.squeeze(tf.one_hot(y_true, depth=y_pred.shape[-1]))

            if self.threshold is None:
                threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
                # make sure [0, 0, 0] doesn't become [1, 1, 1]
                # Use abs(x) > eps, instead of x != 0 to check for zero
                y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-12)
            else:
                y_pred = y_pred > self.threshold

        y_true = tf.cast(y_true, self.dtype)
        y_pred = tf.cast(y_pred, self.dtype)

        def _weighted_sum(val, sample_weight):
            if sample_weight is not None:
                val = tf.math.multiply(val, tf.expand_dims(sample_weight, 1))
            return tf.reduce_sum(val, axis=self.axis)

        self.true_positives.assign_add(_weighted_sum(y_pred * y_true, sample_weight))
        self.false_positives.assign_add(_weighted_sum(y_pred * (1 - y_true), sample_weight))
        self.false_negatives.assign_add(_weighted_sum((1 - y_pred) * y_true, sample_weight))
        self.weights_intermediate.assign_add(_weighted_sum(y_true, sample_weight))
