import numpy as np
from sklearn.model_selection import train_test_split

data = np.load("../datasets/dataset_train_remaining.npz")
specs = data["specs"]
melspecs = data["melspecs"]
cochs = data["cochs"]
targets_linear = data["targets_linear"]
targets_mel = data["targets_mel"]
targets_cochlea = data["targets_cochlea"]

# Take a random 10% for the test set
specs_rest, specs_test, \
melspecs_rest, melspecs_test, \
cochs_rest, cochs_test, \
targets_linear_rest, targets_linear_test, \
targets_mel_rest, targets_mel_test, \
targets_cochlea_rest, targets_cochlea_test = train_test_split(
        specs, melspecs, cochs, targets_linear, targets_mel, targets_cochlea, test_size=0.1, random_state=42
    )

# Take next random 10% for the parameter search set
# Test size = 0.111 because 10% of the original data is ~11.1% of the remaining data
specs_train, specs_param, \
melspecs_train, melspecs_param, \
cochs_train, cochs_param, \
targets_linear_train, targets_linear_param, \
targets_mel_train, targets_mel_param, \
targets_cochlea_train, targets_cochlea_param = train_test_split(
    specs_rest, melspecs_rest, cochs_rest, targets_linear_rest, targets_mel_rest, targets_cochlea_rest,
    test_size=0.111, random_state=42
)

# Save all sets
np.savez("../../datasets/dataset_train.npz", specs=specs_train, melspecs=melspecs_train,
         cochs=cochs_train, targets_linear=targets_linear_train, targets_mel=targets_mel_train, targets_cochlea=targets_cochlea_train)

np.savez("../../datasets/dataset_testset.npz", specs=specs_test, melspecs=melspecs_test,
         cochs=cochs_test, targets_linear=targets_linear_test, targets_mel=targets_mel_test, targets_cochlea=targets_cochlea_test)

np.savez("../../datasets/dataset_param_search.npz", specs=specs_param, melspecs=melspecs_param,
         cochs=cochs_param, targets_linear=targets_linear_param, targets_mel=targets_mel_param, targets_cochlea=targets_cochlea_param)
