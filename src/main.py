import torch # The pytorch lib
import torch.nn as nn # Neural network tools like layers and functions
import torch.optim as optim # Tools for optimizing the model, basically how it learns
from sklearn.model_selection import train_test_split # To split data into training and testing parts
import numpy as np

# Important things when training a model.

# Features, which are the inputs we use to predict. Basically the data to infer.
# Labels, the answers that we're trying to predict. Whether is this or not this.

# It's good to label _train and _test to know which data is used for training and which is for testing.

# if we did:
X = np.random.rand(100, 2) #
# this basically makes a matrix, hypothetically 2 features and 100 rows.
# in each cell the number is either 0 or 1 because of the rand function.

# And if we did:
y = (np.sum(X, axis=1) > 2.5).astype(int)
# for np.sum(X, axis=1). it adds up every row's "features", and the > makes a condition.
# if the output(sum) is greater than 2.5 we mark it as True. Otherwise False.
# the as type int comes in handy because True gets turned into 1 and False into 0.

# And to split the data between testing or training. We can use:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# It basically takes a 0.2 percent out between X and y. Since we have 100 samples
# we'd have 80% (80 samples) training and 20% (20 samples) testing.




