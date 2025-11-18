import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, IPReservoir, Ridge
from reservoirpy.datasets import narma
import matplotlib
from sklearn.metrics import mean_squared_error
matplotlib.use('Qt5Agg')
#rpy.set_seed(52)

data_len = 1000

# Create model
def create_model():
    reservoir = IPReservoir(500, mu=0.0, sigma=0.3, sr=0.9, activation="tanh", epochs=5)
    #reservoir = Reservoir(units=500, sr=0.9, lr=0.2, input_scaling=0.5)
    readout = Ridge(ridge=1e-6, input_dim=500)

    esn = reservoir >> readout

    return reservoir, readout, esn

def create_data():
    X, Y = narma(data_len)

    return X


def pretrain_model(reservoir, X):
    return reservoir.fit(X)

def test_model(esn, x_test):
    return esn.run(x_test)


def evaluate_model(y_pred, y_test):
    mse = mean_squared_error(y_test, y_pred)
    print("Test MSE:", mse)


if __name__ == "__main__":
    # Create model and datasets
    reservoir, readout, esn = create_model()
    X = create_data()


