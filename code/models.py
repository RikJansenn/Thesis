from reservoirpy.nodes import Reservoir, IPReservoir, Ridge
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from reservoirpy.datasets import narma

class ShallowNetwork:
    """
    Shallow ESN with optional IP and Tonotopic mapping
    """

    def __init__(self, N, sr, lr, input_scaling, sigma, ridge, input_dim, IP=True):
        self.N = N
        self.sr = sr
        self.lr = lr
        self.input_scaling = input_scaling
        self.sigma = sigma
        self.ridge = ridge
        self.input_dim = input_dim
        self.IP = IP
        self.workers = 1

        # Create reservoir
        if self.IP:
            self.reservoir = IPReservoir(N, sr=sr, lr=lr, mu=0.0, sigma=sigma, activation="tanh", epochs=4,
                                         learning_rate=3e-4, input_scaling=input_scaling, dtype=np.float32)
        else:
            self.reservoir = Reservoir(N, sr=sr, lr=lr, input_scaling=input_scaling, dtype=np.float32)

        # Create readout
        self.readout = Ridge(ridge=ridge, output_dim=11)

    def create_input_weights(self, p=0.1):
        n = 1 / self.input_dim
        Win = np.random.uniform(0.5 * n, n, (self.reservoir.units, self.input_dim))
        mask = np.random.rand(self.reservoir.units, self.input_dim) < p
        Win *= mask
        self.reservoir.Win = Win
        self.reservoir.input_dim = self.input_dim

    def apply_ip(self, p=0.1):
        self.reservoir.Win = np.random.uniform(0.5, 1, (self.reservoir.units, 1))

        mask = np.random.rand(self.reservoir.units, 1) < p
        self.reservoir.Win *= mask

        T = 1000
        _, X_narma = narma(T)
        _ = self.reservoir.fit(X_narma, warmup=100)

    def train(self, X, Y):
        ### Find readout parameter ###
        # Make parameter search test split
        # Run crossvalidation

        states_list = self.reservoir.run(X, workers=self.workers)

        self.readout.fit(states_list, y=Y, workers=self.workers)

    def test(self, X_test, Y_test):
        states_list = self.reservoir.run(X_test, workers=self.workers)

        y_pred = []
        timestep_predictions = []
        y_true = []

        for X_seq, states, y_seq in zip(X_test, states_list, Y_test):
            predictions = self.readout.run(states)                          # Get raw prediction per timestep
            pred_per_timestep = np.argmax(predictions, axis=1)              # Get one-hot winner at each timestep
            timestep_predictions.append(pred_per_timestep)

            non_silence_preds = pred_per_timestep[pred_per_timestep != 10]  # Remove silence as category
            final_pred = mode(non_silence_preds, keepdims=False).mode       # Get winning digit with majority voting

            y_pred.append(final_pred)

            # Get true labels for this sequence
            y_per_timestep = np.argmax(y_seq, axis=1)
            non_silence_true = y_per_timestep[y_per_timestep != 10]
            y_true.append(non_silence_true[0])

        # Convert to arrays
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        timestep_predictions = list(timestep_predictions)
        y_per_timestep = np.array([np.argmax(y_seq, axis=1) for y_seq in Y_test])

        # Compute accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Test accuracy: {accuracy:.3f}")

        return accuracy, y_true, y_pred, timestep_predictions, y_per_timestep
