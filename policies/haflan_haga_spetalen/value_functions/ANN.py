import tempfile
import time

from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
import tensorflow as tf


class ANN:
    """
    Wrapper class for the keras Sequential model
    """

    def __init__(
        self,
        network_structure,
        input_dimension,
        trace_decay,
        discount_factor,
        learning_rate,
    ):
        self.trace_decay = trace_decay
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.network_structure = network_structure
        self.input_dimension = input_dimension

        self.model = self.create_model()
        self.predict_model = self.create_model()
        self.update_predict_model()

        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"logs/relu_{discount_factor}_{network_structure}_{learning_rate}_{int(time.time())}",
            profile_batch=100000000,  # https://github.com/tensorflow/tensorboard/issues/2819
        )

    def create_model(self):
        model = keras.models.Sequential()
        # First layer needs input size as argument
        nodes_in_first_layer, *nodes_in_rest_of_layers = self.network_structure
        model.add(
            keras.layers.Dense(
                nodes_in_first_layer,
                input_dim=self.input_dimension,
            )
        )
        for layer in nodes_in_rest_of_layers:
            model.add(keras.layers.Dropout(0.5))
            model.add(
                keras.layers.Dense(layer, kernel_regularizer=keras.regularizers.L2())
            )
            model.add(keras.layers.Activation("relu"))
        # The last layer needs to have a single value function output
        model.add(keras.layers.Dense(1))
        optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        model.compile(
            loss="mean_squared_error",
            optimizer=optimizer,
            metrics=["mean_absolute_percentage_error", "mean_absolute_error"],
        )

        return model

    def predict(self, state_features):
        return float(self.predict_model(tf.convert_to_tensor([state_features]))[0][0])

    def update_predict_model(self):
        self.predict_model.set_weights(self.model.get_weights())

    def batch_fit(self, features_list, target_list):
        self.model.fit(
            features_list, target_list, callbacks=[self.tensorboard],
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        # Adding bytes from file to model attribute
        for key, value in state.items():
            if key.__contains__("model"):
                with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
                    keras.models.save_model(state.get(key), fd.name, overwrite=True)
                    state[key] = fd.read()
        state["tensorboard"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for key, value in state.items():
            if key.__contains__("model"):
                with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=True) as fd:
                    fd.write(state[key])
                    fd.flush()
                    setattr(self, key, keras.models.load_model(fd.name))
        self.tensorboard = ModifiedTensorBoard()


class ModifiedTensorBoard(TensorBoard):
    """
    Since normal tensorboard will create a new log file for every .fit call we create a custom Tensorboard class
    """

    def __init__(self):
        super().__init__()
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir
        self._train_dir = self.log_dir
        self._train_step = 0

        # Overriding this method to stop creating default log writer

    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self._train_step += 1
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()
