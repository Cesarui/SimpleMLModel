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

# After we have some data. We need to convert it into Pytorch Tensors. Which is basically their own version
# of arrays.

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# The dtype torch float 32 is "the right kind of float" that Pytorch needs for the math.
# Then the view reshapes the vector into a column shape since Pytorch expects 2D.

# This builds a simple Neural Network
model = nn.Sequential(
    nn.Linear(2,10), # Takes in 2 inputs and connects them to 10 neurons.
    nn.ReLU(), # This works as the activation function
    nn.Linear(10,1), # Takes the 10 neurons and compresses them to one output
    nn.Sigmoid() # And then the output is either 0 or 1
)

# This is how we define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01) # I learned that 0.01 or 0.001 is good for testing,
# but can be tweaked accordingly.

# The criterion measures how far our predictions are from the real labels.
# and the optimizer tells pytorch how to adjust the weights each round, apparently Adam is a popular choice.
# And lr is the learning rate.

# High learning would overshoot and low would be learning slow but carefully.

# Now comes the step to actually train the model
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')

# Each loop is one training Epoch. Which means it passes through all the data at once.
# The order goes as:
# 1. Make Predictions
# 2. Compare with real labels
# 3. Calculate how wrong it was.
# 4. Backpropagate. Which means to figure out which weights caused an error.
# 5. Adjust weights slightly
# 6. Repeat 100 times

# Testing
with torch.no_grad():
    predictions = model(X_test)
    predicted = (predictions > 0.5).float()
    accuracy = (predicted == y_test).float().mean()
    print(f'Accuracy: {accuracy:.4f}')
# The torch no grad turns off the training so no more computing gradients.
# predictions, if 0.7 yes, if 0.2 no
# Compare to actual labels and calculate accuracy.


# Some questions i still have for tomorrow:

