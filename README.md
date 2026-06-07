# MLP

A neural network library in C++, written from scratch to understand how neural
networks learn: from the single perceptron to a multilayer perceptron trained
with backpropagation that reads handwritten digits.

- **Perceptron**: one neuron with a step activation and the perceptron
  learning rule. It learns AND, OR and NOR, but not XOR.
- **Multilayer network**: fully connected layers trained by stochastic
  gradient descent with backpropagation. The gradients are checked against
  finite differences in the unit tests.
- **Step, sigmoid, tanh and linear activations**, chosen per layer.
- **MNIST**: trains a 784-100-10 network that recognises 97.4% of the test
  digits after 10 epochs.
- **Digit recognition** in PGM images of any size, white on black or black on
  white.
- **Trained networks** can be saved to text files and loaded again.
- **Interactive demo** in the terminal: watch the networks learn, in colour.
- **No dependencies**: plain C++98, a Makefile and a C++ compiler are all it
  needs.

## History

This repository holds three generations of the same project. The older ones
are kept in the history and can be browsed from their tags. The logic gate
truth tables in `data/` date from 2012 and are still read by this version.

| Version | What it was |
|---------|-------------|
| **2012**: [`2012-prototypes`](../../tree/2012-prototypes) | First experiments, as Code::Blocks projects written in French: a single neuron learning logic gates with the perceptron rule, then neurons linked together and the start of a Qt interface. |
| **2015**: [`2015-jalgos`](../../tree/2015-jalgos) | Written during an internship at Jalgos: a Qt interface (`master` at the time), then a console program that learns MNIST, built on Eigen and Intel MKL, with multithreading, a batch mode, a validation set and an adaptive learning rate. The branches `batch`, `console`, `benchmarkIterVSRecur` and `experimental` come from that period. |
| **2026**: `master` | This version: rewritten from scratch with no external library, unit tests, and an end to end MNIST demo, from training to recognising digits in images. |

## Contents

1. [Build and run the tests](#1-build-and-run-the-tests)
2. [Try the interactive demo](#2-try-the-interactive-demo)
3. [Perceptron vs multilayer network on logic gates](#3-perceptron-vs-multilayer-network-on-logic-gates)
4. [Train on MNIST](#4-train-on-mnist)
5. [Recognise digits in images](#5-recognise-digits-in-images)
6. [Using the library](#using-the-library)
7. [How it works](#how-it-works)

## 1. Build and run the tests

Requires `make` and a C++ compiler (g++ or clang++).

```sh
make        # builds bin/libmlp.a and the programs in bin/
make test   # runs the unit tests
```

`make test` should end with `All tests passed`.

## 2. Try the interactive demo

`bin/tui` shows the library at work, full screen in the terminal. It needs
nothing more than the rest of the project, not even a curses library: the
screen is drawn with ANSI escape sequences and Unicode block characters.

```sh
make && bin/tui
```

The logic gates demo once XOR is learned, in colour on the screen (here ░ is
an output near 0 and ▓ near 1):

```text
 MLP  Logic gates: perceptron vs multilayer network
 Gate  AND   OR   NOR   XOR   1 when the inputs differ (not linearly separable)
 Output  ░░ 0   ▒▒ 0.5   ▓▓ 1    ● should be 1   ○ should be 0
╭─ Single perceptron ──────────────────╮╭─ Multilayer network 2-3-1 ───────────╮
│ step activation, perceptron rule     ││ sigmoid, backpropagation, rate 0.5   │
│ Epoch 1000  ✗ gave up after 1000     ││ Epoch 4049  ✓ learned                │
│                                      ││                                      │
│ 1 ●▓▓▓▓▓▓▓▓▓▓▓▓○     x1 x2 → out     ││ 1 ●▓▓▓▓▓▒▒░░░░░○     x1 x2 → out     │
│   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓   ○  1  1 → 1 ✗     ││   ▓▓▓▓▓▓▓▓▒▒░░░░   ○  1  1 → 0.04 ✓  │
│   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓   ●  0  1 → 1 ✓     ││   ▓▓▓▓▓▓▓▓▓▓▒▒░░   ●  0  1 → 0.96 ✓  │
│ x2▓▓▓▓▓▓▓▓▓▓▓▓▓▓   ●  1  0 → 1 ✓     ││ x2▓▓▓▓▓▓▓▓▓▓▓▒▒▒   ●  1  0 → 0.98 ✓  │
│   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓   ○  0  0 → 0 ✓     ││   ▒▒▓▓▓▓▓▓▓▓▓▓▓▒   ○  0  0 → 0.03 ✓  │
│   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓                     ││   ░░▒▓▓▓▓▓▓▓▓▓▓▓                     │
│ 0 ○▓▓▓▓▓▓▓▓▓▓▓▓●                     ││ 0 ○░░░▒▓▓▓▓▓▓▓▓●                     │
│   0     x1     1   not separable     ││   0     x1     1                     │
│                                      ││                                      │
│ Error 0.2500                max 0.75 ││ Error 0.0010                max 0.29 │
│                                      ││ ▅▅▄▄▃▁                               │
│ ▁                                    ││ ██████▇▅▁                            │
│ ████████████████████████████████████ ││ █████████▅▃▁▁▁▁                      │
│ epoch 0                         1000 ││ epoch 0                         4049 │
╰──────────────────────────────────────╯╰──────────────────────────────────────╯
 ←→ gate   space pause   r restart   esc back   q quit
```

| Demo               | What it shows |
|--------------------|---------------|
| **Logic gates**    | A perceptron and a 2-3-1 network learn AND, OR, NOR or XOR side by side, with their error curves and a map of their output over the input square. On XOR the map of the perceptron stays cut by a straight line and it gives up, while the map of the network bends around the examples. |
| **MNIST training** | A new network learns MNIST live: progress through the epoch, the image being learned and the outputs of the network for it, the training error and the test accuracy. Choose the size of the hidden layer, the learning rate, the epochs and the number of images first, and save the result to `data/mnist/mnist.mlp`. |
| **MNIST test**     | Browse the 10000 test digits and the drawings of `examples/data/digits` to see what the network answers, right or wrong, and its accuracy over the images viewed. |

The MNIST demos need the images: run `scripts/download_mnist.sh` first (see
[step 4](#4-train-on-mnist)). The test demo uses the network just trained in
the demo, or else the one saved in `data/mnist/mnist.mlp`.

The keys are listed at the bottom of each screen: arrows and Enter to choose,
Space to pause, `r` to restart (logic gates) or to pick a random image (test),
`m` to jump to the next mistake, Esc to go back and `q` to quit. Run it from
the root of the repository, in a terminal of at least 80x24.

## 3. Perceptron vs multilayer network on logic gates

A single perceptron learns any linearly separable function:

```console
$ bin/perceptron data/OU2.txt
...
Learned in 2 epochs
weight 0: 1
weight 1: 1.1
threshold: 0.9

1 1 -> 1 (expected 1)
0 1 -> 1 (expected 1)
1 0 -> 1 (expected 1)
0 0 -> 0 (expected 0)
```

XOR is not linearly separable, so the perceptron fails:

```console
$ bin/perceptron examples/data/xor2.txt
...
Not learned after 1000 epochs: the problem is probably not linearly separable
```

With a hidden layer of 3 neurons, XOR is learned:

```console
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

Usage: `bin/mlp <data-file> [hidden-neurons] [learning-rate] [max-epochs]`.
The format of the data files is described [below](#data-format).

## 4. Train on MNIST

[MNIST](http://yann.lecun.com/exdb/mnist/) has 60000 training and 10000 test
images of handwritten digits, 28x28 pixels each. Download it (about 11 MB)
into `data/mnist`:

```sh
scripts/download_mnist.sh
```

Then train:

```console
$ bin/mnist_train
Loading MNIST from data/mnist...
60000 training images, 10000 test images of 28x28 pixels
Network 784-100-10, learning rate 0.1

Before training: test accuracy 10.10%
Epoch  1: training error 0.0181, test accuracy 93.94%
Epoch  2: training error 0.0091, test accuracy 95.73%
Epoch  3: training error 0.0070, test accuracy 96.19%
Epoch  4: training error 0.0058, test accuracy 96.63%
Epoch  5: training error 0.0049, test accuracy 96.97%
Epoch  6: training error 0.0043, test accuracy 97.34%
Epoch  7: training error 0.0038, test accuracy 97.20%
Epoch  8: training error 0.0035, test accuracy 97.42%
Epoch  9: training error 0.0032, test accuracy 97.45%
Epoch 10: training error 0.0029, test accuracy 97.37%

Network saved to data/mnist/mnist.mlp

Confusion matrix on the test set (rows: digit, columns: prediction)

           0     1     2     3     4     5     6     7     8     9    accuracy
     0   964     0     1     2     0     1     6     3     1     2       98.4%
     1     0  1123     2     4     0     1     2     1     2     0       98.9%
     ...
```

The training error goes down every epoch, and the accuracy on the test images
(which the network never trains on) goes from 10% (chance) to over 97%.
The network is saved to `data/mnist/mnist.mlp` after every epoch, so you can stop
training at any time with Ctrl-C.

| Option          | Default                | Meaning                                       |
|-----------------|------------------------|-----------------------------------------------|
| `--data DIR`    | `data/mnist`           | directory of the MNIST files                  |
| `--hidden N`    | `100`                  | neurons in the hidden layer                   |
| `--epochs N`    | `10`                   | passes over the training images               |
| `--rate R`      | `0.1`                  | learning rate                                 |
| `--limit N`     | all                    | only train on the first N images (quick runs) |
| `--seed N`      | `1`                    | seed of the initial weights and the shuffling |
| `--output FILE` | `data/mnist/mnist.mlp` | where to save the trained network             |

Each epoch goes through the 60000 images, which takes a while. For a quick
check that everything works, `bin/mnist_train --limit 5000 --epochs 3` only
trains on 5000 images and still reaches about 90%.

## 5. Recognise digits in images

`bin/recognize` loads a trained network and classifies images:

```console
$ bin/recognize data/mnist/mnist.mlp examples/data/digits/7.pgm
examples/data/digits/7.pgm

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%##
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%--
                                      %%%%
                                    ==@@++
                                    %%%%
                                  **@@==
                                  %%%%
                                ##@@::
                              ..@@##
                              ##@@..
                            --@@**
                            %%%%
                          ==@@++
                          %%%%
                        ++@@==
                        %%%%
                      **@@--
                    ..@@%%
                    ##@@::

Recognised digit: 7  (scores: 7=1.00 3=0.00 0=0.00)
```

It prints the image as the network sees it, the recognised digit and the
three best scores. Try:

- **The sample drawings**: `bin/recognize data/mnist/mnist.mlp examples/data/digits/*.pgm`.
  These are digits drawn black on white at 150x150 pixels.
- **MNIST test images**: `mkdir -p data/mnist/samples && bin/mnist_export 20 data/mnist/samples`
  writes the first 20 test images as `data/mnist/samples/<index>-<digit>.pgm` (the
  file name gives the right answer). Then run
  `bin/recognize data/mnist/mnist.mlp data/mnist/samples/*.pgm`.
- **Your own handwriting**: draw a digit (paint program, or a photo of
  paper), crop it around the digit and convert it to PGM, for example with
  ImageMagick: `magick my-digit.png my-digit.pgm`. Then run
  `bin/recognize data/mnist/mnist.mlp my-digit.pgm`. Thick strokes work best.

Images can be of any size and either white on black or black on white. Before
classification they are processed the way MNIST was built: the ink is made
white on black, the drawing is cropped and scaled to fit a 20x20 box, and its
centre of mass is moved to the middle of a 28x28 image.

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
mlp::Network copy = mlp::Network::load("xor.mlp");
```

Compile with `-Iinclude` and link with `bin/libmlp.a`.

| Header             | Contents                                                 |
|--------------------|----------------------------------------------------------|
| `mlp/Activation.h` | step, sigmoid, tanh and linear functions with derivatives |
| `mlp/Perceptron.h` | single neuron trained with the perceptron rule           |
| `mlp/Layer.h`      | fully connected layer, forward and backward passes       |
| `mlp/Network.h`    | multilayer network: train, predict, classify, save, load |
| `mlp/Dataset.h`    | training examples and their text format                  |
| `mlp/Mnist.h`      | reader for the MNIST IDX files                           |
| `mlp/Image.h`      | PGM images and MNIST-style normalisation                 |
| `mlp/Random.h`     | portable random generator (same results on any platform) |

### Data format

```
exemples 4; entrees 2;
1/1/1;
0/1/1;
1/0/1;
0/0/0;
```

The header gives the number of examples (`exemples`) and inputs (`entrees`),
optionally followed by `sorties N;` for several outputs. Each example lists its
inputs, then its expected outputs, separated by `/`. `data/` holds the AND
(`ET3.txt`), OR (`OU2.txt`) and NOR (`NON2.txt`) truth tables, and
`examples/data/` the XOR one.

## How it works

Each neuron computes `f(w . x + b)`. Training minimises the squared error
`E = 1/2 sum (y - t)^2` one example at a time:

1. **Forward pass**: every layer computes its outputs and keeps its inputs
   and weighted sums `s`.
2. **Backward pass**: starting from `dE/dy = y - t` at the output, each layer
   computes `delta_j = dE/dy_j * f'(s_j)` and passes `dE/dx_i = sum_j delta_j w_ji`
   to the previous layer. It then updates its weights with
   `w_ji -= rate * delta_j * x_i` and its biases with `b_j -= rate * delta_j`.

The examples are shuffled at each epoch. Initial weights are drawn uniformly
in `[-1/sqrt(n), 1/sqrt(n)]`, where `n` is the number of inputs of the layer.

## Layout

```
include/mlp/    public headers
src/            library sources
examples/       perceptron, mlp, mnist_train, mnist_export, recognize
examples/tui/   interactive terminal demo (bin/tui)
examples/data/  XOR dataset, sample digit drawings
tests/          unit tests (make test)
scripts/        download_mnist.sh
data/           logic gate datasets (2012), MNIST once downloaded
```
