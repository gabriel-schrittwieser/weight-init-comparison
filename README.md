# Weight Initialization Comparison

### Setup

Set up virtual environment:
```shell
python -m venv venv
venv\Scripts\activate.bat
```

Install dependencies:
```shell
pip install -r requirements.txt
```

---

### mlp.py

This is where the neural network architecture is defined. In this case, a basic Multilayer Perceptron (MLP) using ReLU activations.

```py
import torch
import torch.nn as nn
```
<b>PyTorch</b> - Deep Learning library, used for tensors (multi-dimensional arrays) and DL functionalities.
<b>torch.nn</b> - Provides building blocks for defining a neural network (layers, loss functions, ...)


```py
class MLP(nn.Module):
```
Define a new class `MLP` for our model, using `nn.Module` as a subclass. This is the base class for all neural network models in `PyTorch`, allowing us to use built-in functionalities.

```py
def __init__(self, input_size=784, hidden_sizes=[256, 128], output_size=10):
    super(MLP, self).__init__()
```
<b>__init__()</b>: Constructor method of the class, initializes the layers and parameters of the model.

<b>input_size=784</b>: The number of input features. For `MNIST`, each image is `28x28`, which gives 784 features (pixels).

<b>hidden_sizes=[256, 128]</b>: This defines the number of neurons in the hidden layers. We're using two hidden layers, one with 256 neurons and another with 128 neurons.

<b>output_size=10</b>: Since this is a classification task for `MNIST` (10 digit classes), we want the output size to be 10.

<b>super(MLP, self).__init__()</b>: Calls the constructor of `nn.Module`. It’s necessary to initialize the parent class and ensure all PyTorch's model functionalities are set up.

```py
self.hidden_layers = nn.ModuleList()
```
<b>ModuleList</b> is a container for storing layers. If the number of hidden layers can vary, we can add layers dynamically. This gives us flexibility in defining an arbitrary number of hidden layers, instead of manually defining each layer.

```py
in_features = input_size
```
<b>in_features</b> stores the number of input features. It will be used to define the first layer's input size and update it as hidden layers are added.

```py
# Create hidden layers
for hidden_size in hidden_sizes:
    self.hidden_layers.append(nn.Linear(in_features, hidden_size))
    in_features = hidden_size
```
Loop over the <b>hidden_sizes</b> list, which contains the number of neurons for each hidden layer. This allows us to define the number of layers and their sizes dynamically.  
`nn.Linear` creates a <b>fully connected (dense) layer</b>. It takes a certain number of input features (`in_features`) and outputs a certain number of neurons (`hidden_size`). We append each fully connected layer to the `hidden_layers` list.  
After creating each layer, we update the number of input features for the next layer. The output of each hidden layer becomes the input to the next layer.

```py
# Output layer
self.output = nn.Linear(in_features, output_size)
```
After all hidden layers, we add the output layer. (`nn.Linear` to define the fully connected layer). This layer has `in_features` (the number of neurons in the last hidden layer) as input and `output_size` (10 for `MNIST`) as output.

```py
# Activation function (ReLU)
self.relu = nn.ReLU()
```
<b>ReLU</b> (Rectified Linear Unit) - non-linear activation function, defined here so it can be applied to each hidden layer during the forward pass.

```py
def forward(self, x):
    x = x.view(x.size(0), -1)  # Flatten input from (batch, 1, 28, 28) → (batch, 784)
```
This is where the data actually passes through the model.  
<b>Flattening the input:</b>  
The `MNIST` images are `28x28` pixels, so they have a shape of (`batch_size, 1, 28, 28`). But we need them in a 1D vector format (`[batch_size, 784]`) to input them into a fully connected layer.

`x.view(x.size(0), -1)` flattens the input, keeping the batch size unchanged (`x.size(0)`) and collapsing the `28x28` image into a 784-dimensional vector.

```py
for layer in self.hidden_layers:
    x = self.relu(layer(x))
```
We loop through the hidden layers, for each hidden layer in `self.hidden_layers`, we pass the data `x` through the layer, followed by a ReLU activation.  
`self.relu(layer(x))`: The data is first passed through the linear layer (`layer(x)`), then the ReLU activation is applied to the output of that layer.

```py
x = self.output(x)
```
<b>Final Output</b>: After all hidden layers, the data is passed through the <b>output layer</b> (`self.output(x)`), which produces the final predictions (i.e., a 10-dimensional vector for classification).

#### `mlp-py` summary:
In summary, this code builds a basic feedforward neural network (MLP) with:

- Multiple hidden layers defined dynamically based on the `hidden_sizes` list.
- Each layer is fully connected (`nn.Linear`).
- ReLU activations are used in all hidden layers.
- The output layer is designed for 10 classes (since we use MNIST classification).