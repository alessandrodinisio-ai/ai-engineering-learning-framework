# Multi-Layer Networks and Forward Propagation

> A single neuron draws a line. Stack them up, and you can draw anything.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 01 (Math Foundations), Lesson 03.01 (Perceptron)
**Time:** ~90 minutes

## Learning Objectives

- Build a multi-layer network from scratch using Layer and Network classes to complete a full forward pass
- Trace matrix dimensions through each layer of the network and identify shape mismatches
- Explain why stacking nonlinear activations enables networks to learn curved decision boundaries
- Solve XOR with a 2-2-1 architecture using hand-tuned sigmoid weights

## The Problem

A single neuron is a line-drawer. That's it. One straight line through your data. Every real problem in AI — image recognition, language understanding, playing Go — requires curves. Stacking neurons into multiple layers is how you get curves.

In 1969, Minsky and Papert proved this limitation was fatal: a single-layer network cannot learn XOR. Not "struggles to learn" — mathematically impossible. XOR's truth table puts [0,1] and [1,0] on one side, [0,0] and [1,1] on the other. No single straight line can separate them.

This froze neural network research funding for over a decade. In hindsight, the fix was obvious: stop using one layer. Stack neurons into multiple layers. Let the first layer carve the input space into new features, then let the second layer combine those features into decisions that no single line could make.

That stack is the multi-layer network. It's the foundation of every deep learning model in production today. Forward propagation — data flowing from input through hidden layers to output — is the first thing you must build before anything else can work.

## The Concept

### Layers: Input, Hidden, Output

A multi-layer network has three types of layers:

**Input layer** — not really a layer. It just holds your raw data. Two features means two input nodes. No computation happens here.

**Hidden layer** — where the real work happens. Each neuron takes all outputs from the previous layer, applies weights and a bias, then passes the result through an activation function. Called "hidden" because you never directly observe these values in the training data.

**Output layer** — the final answer. Binary classification uses one neuron with sigmoid. Multi-class uses one neuron per class.

```mermaid
graph LR
    subgraph Input["Input Layer"]
        x1["x1"]
        x2["x2"]
    end
    subgraph Hidden["Hidden Layer (3 neurons)"]
        h1["h1"]
        h2["h2"]
        h3["h3"]
    end
    subgraph Output["Output Layer"]
        y["y"]
    end
    x1 --> h1
    x1 --> h2
    x1 --> h3
    x2 --> h1
    x2 --> h2
    x2 --> h3
    h1 --> y
    h2 --> y
    h3 --> y
```

This is a 2-3-1 network. Two inputs, three hidden neurons, one output. Every connection carries a weight. Every neuron (except input) carries a bias.

Each layer produces a vector of numbers called the hidden state. For text, hidden states grow in dimension — encoding a word as 768 numbers to capture semantics. For images, they reduce dimension — compressing millions of pixels into a manageable representation. The hidden state is where learning actually happens.

### Neurons and Activation

Each neuron does three things:

1. Multiplies each input by its corresponding weight
2. Sums all products and adds a bias
3. Passes this sum through an activation function

For now, the activation function is sigmoid:

```
sigmoid(z) = 1 / (1 + e^(-z))
```

Sigmoid squashes any number into the (0, 1) range. Large positive inputs push toward 1, large negative inputs push toward 0, zero maps to 0.5. This smooth curve is what makes learning possible — unlike the perceptron's hard step, sigmoid has a gradient everywhere.

### Forward Propagation: How Data Flows

Forward propagation pushes input data through the network, one layer at a time, until it reaches the output. No learning happens during forward propagation. It's pure computation: multiply, add, activate, repeat.

```mermaid
graph TD
    X["Input: [x1, x2]"] --> WH["Multiply by weight matrix W1 (2x3)"]
    WH --> BH["Add bias vector b1 (3,)"]
    BH --> AH["Apply sigmoid element-wise"]
    AH --> H["Hidden output: [h1, h2, h3]"]
    H --> WO["Multiply by weight matrix W2 (3x1)"]
    WO --> BO["Add bias vector b2 (1,)"]
    BO --> AO["Apply sigmoid"]
    AO --> Y["Output: y"]
```

At each layer, three operations happen in sequence:

```
z = W * input + b       (linear transformation)
a = sigmoid(z)           (activation)
```

The output of one layer becomes the input to the next. That's the entire forward pass.

### Matrix Dimensions

Tracking dimensions is the single most important debugging skill in deep learning. Here's that 2-3-1 network:

| Step | Operation | Dimensions | Result Shape |
|------|-----------|------------|-------------|
| Input | x | -- | (2,) |
| Hidden linear | W1 * x + b1 | W1: (3, 2), b1: (3,) | (3,) |
| Hidden activation | sigmoid(z1) | -- | (3,) |
| Output linear | W2 * h + b2 | W2: (1, 3), b2: (1,) | (1,) |
| Output activation | sigmoid(z2) | -- | (1,) |

The rule: the weight matrix W for layer k has shape (neurons_in_layer_k, neurons_in_layer_k_minus_1). Rows correspond to the current layer, columns to the previous layer. If the shapes don't match, you have a bug.

### Universal Approximation Theorem

In 1989, George Cybenko proved something remarkable: a neural network with just a single hidden layer and enough neurons can approximate any continuous function to arbitrary precision.

This doesn't mean a single hidden layer is always best. It means this architecture theoretically has the capacity. In practice, deeper networks (more layers, fewer neurons per layer) can learn the same function with far fewer total parameters than a wide, shallow network. That's why deep learning works.

The intuition: each neuron in the hidden layer learns a "bump" or a feature. Enough bumps in the right places can approximate any smooth curve. More neurons, more bumps, better approximation.

```mermaid
graph LR
    subgraph FewNeurons["4 Hidden Neurons"]
        A["Rough approximation"]
    end
    subgraph MoreNeurons["16 Hidden Neurons"]
        B["Close approximation"]
    end
    subgraph ManyNeurons["64 Hidden Neurons"]
        C["Near-perfect fit"]
    end
    FewNeurons --> MoreNeurons --> ManyNeurons
```

### Composability

Neural networks are composable. You can stack them, chain them, run them in parallel. A Whisper model uses one encoder network for audio and a separate decoder network to produce text. Modern LLMs are decoder-only. BERT is encoder-only. T5 is encoder-decoder. The architecture choice determines what the model can do.

## Build It

Pure Python, no numpy. Every matrix operation hand-written from scratch.

### Step 1: Sigmoid Activation

```python
import math

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))
```

Clamping to [-500, 500] prevents overflow. `math.exp(500)` is large but finite; `math.exp(1000)` is infinity.

### Step 2: Layer Class

The single most important operation in all of deep learning is matrix multiplication. Every layer, every attention head, every forward pass — matrix multiply from start to finish. A linear layer takes an input vector, multiplies it by a weight matrix, and adds a bias vector: y = Wx + b. This one equation accounts for 90% of the compute in neural networks.

A layer holds a weight matrix and a bias vector. Its forward method takes an input vector and returns the activated output.

```python
class Layer:
    def __init__(self, n_inputs, n_neurons, weights=None, biases=None):
        if weights is not None:
            self.weights = weights
        else:
            import random
            self.weights = [
                [random.uniform(-1, 1) for _ in range(n_inputs)]
                for _ in range(n_neurons)
            ]
        if biases is not None:
            self.biases = biases
        else:
            self.biases = [0.0] * n_neurons

    def forward(self, inputs):
        self.last_input = inputs
        self.last_output = []
        for neuron_idx in range(len(self.weights)):
            z = sum(
                w * x for w, x in zip(self.weights[neuron_idx], inputs)
            )
            z += self.biases[neuron_idx]
            self.last_output.append(sigmoid(z))
        return self.last_output
```

The weight matrix has shape (n_neurons, n_inputs). Each row is one neuron's weights for all inputs. The forward method iterates over neurons, computes the weighted sum plus bias, applies sigmoid, and collects the results.

### Step 3: Network Class

A network is a list of layers. Forward propagation chains them: layer k's output feeds layer k+1.

```python
class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current
```

That's the entire forward pass. Four lines of logic. Data goes in, flows through each layer, comes out the other end.

### Step 4: Hand-Tuned Weights for XOR

In Lesson 01, we solved XOR by combining OR, NAND, and AND perceptrons. Now we do the same with our Layer and Network classes. 2-2-1 architecture: two inputs, two hidden neurons, one output.

```python
hidden = Layer(
    n_inputs=2,
    n_neurons=2,
    weights=[[20.0, 20.0], [-20.0, -20.0]],
    biases=[-10.0, 30.0],
)

output = Layer(
    n_inputs=2,
    n_neurons=1,
    weights=[[20.0, 20.0]],
    biases=[-30.0],
)

xor_net = Network([hidden, output])

xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

for inputs, expected in xor_data:
    result = xor_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    print(f"  {inputs} -> {result[0]:.6f} (rounded: {predicted}, expected: {expected})")
```

Large weights (20, -20) make sigmoid behave like a step function. The first hidden neuron approximates OR, the second approximates NAND, and the output neuron combines them as AND — producing XOR overall.

### Step 5: Circle Classification

A harder problem: classify 2D points as inside or outside a circle centered at the origin with radius 0.5. This requires a curved decision boundary — something a single perceptron cannot do.

```python
import random
import math

random.seed(42)

data = []
for _ in range(200):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    label = 1 if (x * x + y * y) < 0.25 else 0
    data.append(([x, y], label))

circle_net = Network([
    Layer(n_inputs=2, n_neurons=8),
    Layer(n_inputs=8, n_neurons=1),
])
```

With random weights, the network won't classify well. But forward propagation runs just fine. That's the point — forward propagation is just computation. Learning the correct weights requires backpropagation, which is the topic of Lesson 03.

```python
correct = 0
for inputs, expected in data:
    result = circle_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    if predicted == expected:
        correct += 1

print(f"Accuracy with random weights: {correct}/{len(data)} ({100*correct/len(data):.1f}%)")
```

Random weights give poor accuracy — often worse than just guessing the majority class. After training (Lesson 03), the same 8-hidden-neuron architecture will draw a curved boundary separating inside from outside.

## Use It

PyTorch does all of the above in four lines:

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Sigmoid(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
output = model(x)
print(output)
```

`nn.Linear(2, 8)` is your Layer class: weight matrix of shape (8, 2), bias vector of shape (8,). `nn.Sigmoid()` is your element-wise sigmoid function. `nn.Sequential` is your Network class: chaining layers in order.

The difference is speed and scale. PyTorch runs on GPUs, handles batches of millions of samples, and automatically computes gradients for backpropagation. But the forward pass logic is identical to what you just built from scratch.

## Ship It

This lesson produces a reusable prompt for designing network architectures:

- `outputs/prompt-network-architect.md`

Use it when you need to decide how many layers, how many neurons per layer, and which activation functions to use for a given problem.

## Exercises

1. Build a 2-4-2-1 network (two hidden layers) and run a forward pass on XOR data with random weights. Print the intermediate hidden layer outputs to see how the representation transforms at each layer.

2. Change the hidden layer size in the circle classifier from 8 to 2, then to 32. Run forward propagation with random weights each time. Does the number of hidden neurons change the range or distribution of outputs? Why?

3. Implement a `count_parameters` method on the Network class that returns the total number of trainable weights and biases. Test it on a 784-256-128-10 network (classic MNIST architecture). How many parameters does it have?

4. Build a forward pass for a 3-4-4-2 network. Feed it RGB color values (normalized to 0-1) and observe the two outputs. This is the architecture of a simple binary color classifier.

5. Replace sigmoid with a "leaky step" function: return 0.01 * z when z < 0, otherwise return 1.0. Run forward propagation on XOR using the same hand-tuned weights from Step 4. Does it still work? Why is the smooth sigmoid preferred over hard cutoffs?

## Key Terms

| Term | What People Say | What It Actually Is |
|------|----------------|----------------------|
| Forward pass | "Running the model" | Pushing input through each layer — multiply weights, add bias, activate — to produce an output |
| Hidden layer | "The middle part" | Any layer between input and output whose values are never directly observed in the data |
| Multi-layer network | "A deep neural network" | Neurons arranged in sequential layers, each layer's output feeding the next layer's input |
| Activation function | "The nonlinearity" | A function applied after the linear transformation to introduce curves into decision boundaries |
| Sigmoid | "The S-curve" | sigma(z) = 1/(1+e^(-z)), squashes any real number to (0,1), smooth and differentiable everywhere |
| Weight matrix | "The parameters" | A matrix W of shape (current_layer_neurons, previous_layer_neurons) holding learnable connection strengths |
| Bias vector | "The offset" | A vector added after matrix multiplication, allowing neurons to activate even when all inputs are zero |
| Universal approximation | "Neural nets can learn anything" | A single hidden layer with enough neurons can approximate any continuous function — but "enough" may mean billions |
| Linear transformation | "The matrix multiply step" | z = W * x + b, the computation before activation that maps input to a new space |
| Decision boundary | "Where the classifier switches" | The surface in input space where the network output crosses the classification threshold |

## Further Reading

- Michael Nielsen, *Neural Networks and Deep Learning* Ch. 1-2 (http://neuralnetworksanddeeplearning.com/) — The clearest free explanation of forward propagation and network structure, with interactive visualizations
- Cybenko, *Approximation by Superpositions of a Sigmoidal Function* (1989) — The original universal approximation theorem paper, surprisingly readable
- 3Blue1Brown, *But what is a neural network?* (https://www.youtube.com/watch?v=aircAruvnKk) — 20-minute visual explanation of layers, weights, and forward propagation that builds correct mental models
- Goodfellow, Bengio, Courville, *Deep Learning* Ch. 6 (https://www.deeplearningbook.org/) — Standard reference for multi-layer networks, free online
