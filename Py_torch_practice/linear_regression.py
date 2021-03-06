import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        #Looks like this is the first layer
        #and the only layer 
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out

x_values = [i for i in range(10)]

x_train = np.array(x_values, dtype=np.float32)

print(x_train)
x_train =x_train.reshape(5,2)

print(x_train)
y_values = [4*i +1 for i in range(5) ]

y_train = np.array(y_values, dtype=np.float32)

y_train = y_train.reshape(5,1)

print(y_train.shape[1])
print(y_train)
input_size = 2 ## this is the number of x values i.e y = x0 + x1m +x2m
output_size = 1 # dependent variable. only one y per function of y = x0 + x1m....

model = LinearRegressionModel(input_size, output_size)

evaluater = nn.MSELoss()

learning_rate = 0.01

weight_updater = torch.optim.SGD(model.parameters(),lr=learning_rate)

inputs = Variable(torch.from_numpy(x_train))
GT = Variable(torch.from_numpy(y_train))


for epoch in range(100):

    weight_updater.zero_grad()

    outputs = model(inputs)
    loss = evaluater(outputs, GT)

    loss.backward()

    weight_updater.step()

    print("epoch {}, loss {}".format(epoch, loss.data[0]))


#lets get predicted values
predictions = model(Variable(torch.from_numpy(x_train))).data.numpy()

print(predictions)