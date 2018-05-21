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

'''
these x values have 2 dimensions that are fed.  These would 
be the x values in y = x0 + x1m1 + x2m2....

Since we are predicting 11 classes this needs to also be
11 rows long
'''
x_values = [[i,i] for i in range(11)]

'''
the cross entropy loss function takes a float32 tensor and
a long tensor.  Float is for the x_values that will be passed
to the linear regression model and then passed to 
the loss function.
'''
x_train = np.array(x_values, dtype=np.float32)
labels = Variable(torch.LongTensor([0,1,2,3,4,5,6,7,8,9,10]))

'''
input size is how many x values we are saying will come
y = x0 + x1m1 + x2m2.....

the x values being the amount of x's supplied.  However I 
do not believe my example is completely accurate of what's
going on in this case.  

The output size is the amount of predictions to be made.  This
must be the same amount of classes since the cross entropy
function will give a probability for each class. 
'''
input_size = 2
output_size = 11

model = LinearRegressionModel(input_size, output_size)
evaluater = nn.CrossEntropyLoss()

learning_rate = 0.01
weight_updater = torch.optim.SGD(model.parameters(),lr=learning_rate)
    
'''
Since we are using a Variable class from autograd, it looks
like we are passing nothing to other parts of the code.  This
is true for us programmers just using pytorch.  However, 
the data is actually being passed to each other by way of 
a graph being made in the background.  Hence the name autograd,
it automatically builds graphs and gradients as long as we 
state which variables should be on it.
'''
inputs = Variable(torch.from_numpy(x_train)) 

for epoch in range(100):

    weight_updater.zero_grad()

    outputs = model(inputs)

    print(outputs)
    loss = evaluater(outputs, labels)

    loss.backward()
    
    '''
    Right here is what I was touching on about the Variables.

    How does this thing update the weights if they are never
    supplied to it?  Because we are using the graph and this
    already has reference to those weights on that graph when
    it was created.
    '''
    weight_updater.step()

    print("epoch {}, loss {}".format(epoch, loss.data[0]))

print(x_train)

predictions = model(Variable(torch.from_numpy(x_train))).data.numpy()


'''
These predictions will look crazy. However the position of the
highest number is the what the model predicted the class to be
for the given 2 x values in out x_train list.  which was 11 x 2
11x2 being that there are 11 entries of 2 x values.

There will be 11 predictions for each row in the 11x2 matrix

Choose the highest of the 11 predictions and the label at that
position is the real prediction by the model

Continue to look at the other 10 OUTPUTS and each of there
11 predictions
'''
print(predictions)

'''
an example of another inference prediction when supplied
with 2 x values.  The model I believe is off by about 2.5
since the loss is also around 2.5.  This should have said 
it was closer to 3rd or 2nd position however it was 
highest at the 6 position.

This is saying that given x values of [1,2] the prediction
is 5.

however this was trained saying that x values of [5,5] 
should be 5.

This is showing that the model is poor at predicting this 
label
'''
print(model(Variable(torch.Tensor([1,2]))).data.numpy())