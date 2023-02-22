import logging
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


class EarlyStoppingWithWeightF1(tf.keras.callbacks.EarlyStopping):
    def __init__(self, **kwargs):

        self.config = kwargs.pop('config')
        super().__init__(**kwargs)

    def get_monitor_value(self, logs):
        logs = logs or {}
        if "weighted_val_f1_score" in self.monitor:
            weighted_f1, _ = get_weighed_f1(config=self.config, logs=logs)
            logs[self.monitor] = weighted_f1
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(f"Early stopping conditioned on metric {self.monitor} which is not available.\n"
                            f"Available metrics are: {','.join(list(logs.keys()))}")
        return monitor_value

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

        print('Best F1 for now is {}'.format(self.best))

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
        if self.restore_best_weights:
            if self.verbose > 0:
                print('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)


class PrintWeightF1(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):

        super().__init__()
        self.config = kwargs.pop('config')

    def on_epoch_end(self, epoch, logs=None):
        weighted_f1, _ = get_weighed_f1(config=self.config, logs=logs)
        print('Weighted F1 for epoch {} is {}'.format(epoch + 1, weighted_f1))


class WeightF1(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):

        super().__init__()
        self.config = kwargs.pop('config')

    def on_epoch_end(self, epoch, logs=None):
        weighted_f1, val_weighted_f1 = get_weighed_f1(config=self.config, logs=logs)
        logs["weighted_f1"] = weighted_f1
        logs["val_weighted_f1"] = val_weighted_f1


class F1Score(tf.keras.metrics.Metric):
    __doc__ = "结合tf.keras.metric的计算原理实现precision,recall,f1的metrics"

    def __init__(self, thresholds=0.5, name='f1', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')
        self.thresholds = thresholds

    def update_state(self, y_true, y_pred, sample_weight=None):
        min_delta = 1e-6
        y_pred = tf.cast(tf.where(y_pred > self.thresholds, 1, 0), tf.int8)
        y_true = tf.cast(y_true, tf.int8)

        tp = tf.math.count_nonzero(y_pred * y_true, dtype=tf.float32)
        fp = tf.math.count_nonzero(y_pred * (1 - y_true), dtype=tf.float32)
        fn = tf.math.count_nonzero((1 - y_pred) * y_true, dtype=tf.float32)

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

        self.f1.assign(2 * self.tp / (2 * self.tp + self.fp + self.fn + min_delta))

    def result(self):
        return self.f1

    def reset_state(self):
        # The state of the metric will be reset at the start of each epoch.
        self.f1.assign(0.)
        self.tp.assign(0.)
        self.fp.assign(0.)
        self.fn.assign(0.)


class PrintBest(tf.keras.callbacks.Callback):
    def __init__(self, monitor, mode="max"):

        super().__init__()
        self.monitor = monitor
        self.best = 0

        if mode == "max":
            self.monitor_op = np.greater
        else:
            self.monitor_op = np.less

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(f"PrintBest conditioned on metric {self.monitor} which is not available.\n" f"Available metrics are: {','.join(list(logs.keys()))}")

        return monitor_value

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current, self.best):
            self.best = current
        print(f'Best {self.monitor} until now is {self.best}')


def get_weighed_f1(config, logs):

    logs = logs or {}
    sum_f1 = 0
    val_sum_f1 = 0
    sum_weight = 0
    if len(config.targets) == 1:
        return logs.get("val_f1_score", 0)
    for target in config.targets:
        f1_weight = config.feeds[target].save_f1_weight
        metric = "{}_f1_score".format(target)
        sum_f1 += logs.get(metric, 0) * f1_weight
        val_metric = "val_{}_f1_score".format(target)
        val_sum_f1 += logs.get(val_metric, 0) * f1_weight
        sum_weight += f1_weight
    val_weighted_f1 = val_sum_f1 / sum_weight
    weighted_f1 = sum_f1 / sum_weight

    return weighted_f1, val_weighted_f1


class Logger(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch : {epoch+1}")

        for i in logs:
            print(f"    {i:<20} = {logs[i]}")
        print("\n\n")


common_callbacks = [
    Logger(),
    # Tensorboard in k8s core dump incidentally, disable in product environment.
    # tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    tf.keras.callbacks.CSVLogger('training.csv')
]


class PrintCRFParams(tf.keras.callbacks.Callback):
    __doc__ = "Add print crf trans params."

    def __init__(self, chain_kernel_params):
        super().__init__()
        self.chain_kernel_params = chain_kernel_params

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(self.chain_kernel_params)
        print("\n")
        print("crf trans params:")
        print(trans)
