# MLP

A small neural network library in C++, written to understand how neural
networks learn: from the single perceptron to the multilayer perceptron
trained with backpropagation.

- `Perceptron`: single neuron with a step activation and the perceptron
  learning rule. Learns AND, OR, NOR... but not XOR.
- `Network`: multilayer perceptron made of fully connected `Layer`s, trained
  by stochastic gradient descent with backpropagation. Learns XOR with one
  hidden layer.
- Step, sigmoid, tanh and linear activations, chosen per layer.
- `Dataset`: training examples read from a small text format.
- Networks can be saved to and loaded from text files.
- Plain C++98, no dependencies.

## Build

```sh
make        # bin/libmlp.a and the example programs
make test   # run the unit tests
```

## Examples

```sh
$ bin/perceptron data/OU2.txt
...
Learned in 2 epochs
weight 0: 1
weight 1: 1.1
threshold: 0.9

$ bin/perceptron examples/data/xor2.txt
...
Not learned after 1000 epochs: the problem is probably not linearly separable

$ bin/mlp examples/data/xor2.txt
Network 2-3-1, learning rate 0.5

epoch 1000: error 0.0880478
epoch 2000: error 0.00380189
epoch 3000: error 0.00163759
epoch 4000: error 0.00101919
epoch 4050: error 0.000999948

Learned in 4050 epochs

1 1 -> 0.0383631 (expected 0)
0 1 -> 0.964503 (expected 1)
1 0 -> 0.976582 (expected 1)
0 0 -> 0.0268267 (expected 0)
```

## Using the library

```cpp
#include "mlp/Dataset.h"
#include "mlp/Network.h"

mlp::Dataset data = mlp::Dataset::load("examples/data/xor2.txt");

mlp::Network net(2);                         // 2 inputs
net.addLayer(3, mlp::Activation(mlp::TANH)); // hidden layer
net.addLayer(1);                             // sigmoid output

mlp::TrainingOptions options;
options.learningRate = 0.5;
options.targetError = 0.001;
mlp::TrainingResult result = net.train(data, options);

std::vector<double> output = net.predict(data[0].inputs);
net.save("xor.mlp");
```

Compile with `-Iinclude` and link with `bin/libmlp.a`.

## How it works

Each neuron computes `f(w . x + b)`. Training minimises the squared error
`E = 1/2 sum (y - t)^2` one example at a time:

1. **Forward pass**: every layer computes its outputs and keeps its inputs
   and weighted sums `s`.
2. **Backward pass**: starting from `dE/dy = y - t` at the output, each layer
   computes `delta_j = dE/dy_j * f'(s_j)`, passes `dE/dx_i = sum_j delta_j w_ji`
   to the previous layer, then updates `w_ji -= rate * delta_j * x_i` and
   `b_j -= rate * delta_j`.

The tests check these gradients against finite differences.

## Data format

```
exemples 4; entrees 2;
1/1/1;
0/1/1;
1/0/1;
0/0/0;
```

The header gives the number of examples (`exemples`) and inputs (`entrees`),
optionally followed by `sorties N;` for several outputs. Each example lists its
inputs then its expected outputs, separated by `/`. `data/` holds the AND
(`ET3.txt`), OR (`OU2.txt`) and NOR (`NON2.txt`) truth tables, and
`examples/data/` the XOR one.

## Layout

```
include/mlp/   public headers
src/           library sources
examples/      perceptron and mlp command line programs
tests/         unit tests (make test)
data/          logic gate datasets
```
