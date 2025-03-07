import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Embedding, Dot, Add, Flatten
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


class NeuralLatentFactor:
    """
    Neural Latent Factorization Model.
    """

    def __init__(self, training_df, validation_df, K=10, reg=0.0, epochs=5, batch_size=128, learning_rate=0.01):
        assert isinstance(training_df, pd.DataFrame) and isinstance(validation_df, pd.DataFrame), \
            "training_df and validation_df must be pandas DataFrames."

        self.training_df = training_df
        self.validation_df = validation_df
        self.K = K
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.mu = training_df.power_z.mean()
        self.N = training_df.periodId.max() + 1  # number of periods
        self.M = training_df.farmId.max() + 1  # number of farms
        self.model = None

    def build_model(self):
        """
        Build the matrix factorization model using Keras.
        """
        try:
            u = Input(shape=(1,))
            p = Input(shape=(1,))
            u_embedding = Embedding(self.N, self.K, embeddings_regularizer=l2(self.reg))(u)
            p_embedding = Embedding(self.M, self.K, embeddings_regularizer=l2(self.reg))(p)
            u_bias = Embedding(self.N, 1, embeddings_regularizer=l2(self.reg))(u)
            p_bias = Embedding(self.M, 1, embeddings_regularizer=l2(self.reg))(p)
            x = Dot(axes=2)([u_embedding, p_embedding])
            x = Add()([x, u_bias, p_bias])
            x = Flatten()(x)
            self.model = Model(inputs=[u, p], outputs=x)
            self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate), metrics=['mse'])
        except Exception as e:
            print(f"Error building model: {e}")

    def train(self):
        """
        Train the matrix factorization model using the training data.
        """
        try:
            history = self.model.fit(
                x=[self.training_df.periodId.values, self.training_df.farmId.values],
                y=self.training_df.power_z.values - self.mu,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(
                    [self.validation_df.periodId.values, self.validation_df.farmId.values],
                    self.validation_df.power_z.values - self.mu
                )
            )
            return history
        except Exception as e:
            print(f"Error during training: {e}")

    def plot_losses(self, history):
        """
        Plot the training and validation losses.
        """
        try:
            plt.plot(history.history['loss'], label="train loss")
            plt.plot(history.history['val_loss'], label="test loss")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Error plotting losses: {e}")

    def plot_mse(self, history):
        """
        Plot the training and validation MSE.
        """
        try:
            plt.plot(history.history['mse'], label="train mse")
            plt.plot(history.history['val_mse'], label="test mse")
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Error plotting MSE: {e}")

    def predict_validation_data(self, test_df):
        """
        Predict the validation data using the trained model.
        """
        try:
            validation_predictions = self.model.predict(
                [test_df.periodId.values, test_df.farmId.values]
            ) + self.mu
            return validation_predictions
        except Exception as e:
            print(f"Error during validation prediction: {e}")