import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, IPReservoir, Ridge
from reservoirpy.datasets import narma
import matplotlib
from sklearn.metrics import mean_squared_error
matplotlib.use('Qt5Agg')
#rpy.set_seed(52)

data_len = 5000
train_len = 4000

# Create model
def create_model():
    reservoir = IPReservoir(500, mu=0.0, sigma=0.3, sr=0.9, activation="tanh", epochs=5)
    #reservoir = Reservoir(units=500, sr=0.9, lr=0.2, input_scaling=0.5)
    readout = Ridge(ridge=1e-6, input_dim=500)

    esn = reservoir >> readout

    return reservoir, readout, esn

def create_data():
    X, Y = narma(data_len)
    X = X[-len(Y):]

    X_train, Y_train = X[:train_len], Y[:train_len]
    X_test, Y_test = X[train_len:], Y[train_len:]

    return X_train, Y_train, X_test, Y_test


def train_model(x_train, y_train):
    return esn.fit(x_train, y_train)


def test_model(esn, x_test):
    return esn.run(x_test)


def evaluate_model(y_pred, y_test):
    mse = mean_squared_error(y_test, y_pred)
    print("Test MSE:", mse)


if __name__ == "__main__":

    # Create model and datasets
    reservoir, readout, esn = create_model()
    X_train, Y_train, X_test, Y_test = create_data()

    # Train model
    esn = train_model(X_train, Y_train)

    # Run and evaluate model
    y_pred = test_model(esn, X_test)
    evaluate_model(y_pred, Y_test)
