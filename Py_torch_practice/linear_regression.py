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
        #and the only layer other than an
        #output layer which manipulates the output
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out

x_values = [i for i in range(11)]

x_train = np.array(x_values, dtype=np.float32)

x_train =x_train.reshape(-1,1)

y_values = [2*i +1 for i in x_values ]

y_train = np.array(y_values, dtype=np.float32)

y_train = y_train.reshape(-1,1)
input_size = 1
output_size = 1

model = LinearRegressionModel(input_size, output_size)

evaluater = nn.MSELoss()

learning_rate = 0.01

weight_updater = torch.optim.SGD(model.parameters(),lr=learning_rate)

for epoch in range(100):
    
    inputs = Variable(torch.from_numpy(x_train))
    GT = Variable(torch.from_numpy(y_train))

    weight_updater.zero_grad()

    outputs = model(inputs)
    loss = evaluater(outputs, GT)

    loss.backward()

    weight_updater.step()

    print("epoch {}, loss {}".format(epoch, loss.data[0]))


#lets get predicted values
predictions = model(Variable(torch.from_numpy(x_train))).data.numpy()

print(predictions)