# Neural Network
A barebones implementation of a rudimentary artifical neural network. This was inspired by the 3Blue1Brown [YouTube series on neural networks](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi). The stochastic training idea came from Nielsen's [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com). Currently, a test has been setup training the network on the MNIST handwritten digit database.

## How it works
The implementation only relies on NumPy for vector and matrix calculations; the core logic is built-up from scratch. This invariably means that all computations are done by the CPU.

The network has three components:
- activations: values in $[0,1]$ representing the activation of a perceptron;
- biases: the constant-valued contribution to the activation and the first of two attributes which are calibrated during training;
- weights: the linear factor contribution to the activation and the other attribute to be calibrated.

Activations are computed as follows: 
$$a^{(i+1)} = \sigma(w^{(i)}a^{(i)} + b^{(i)})$$

I have sacrificed elaborate code for a cleaner-looking and more mathematical approach. For example, the partial derivatives of the cost function $C(a^{(L)}_0,\ldots,a^{(L)}_k) = \sum_{i = 0}^{k}(y_k - a^{(L)}_k)$ (where $a^{(L)}$ is the activation in the last layer) with respect to the weights matrix $w^{(i)}$—which represents the weights between the $i$-th and $(i+1)$-th layers—is implemented as:
```python
dc_dw[i] = np.dot(dc_db[i], self.activations[i].transpose())
```
This corresponds to:
$$\frac{\partial C}{\partial w^{(i)}} = a^{(i)}\sigma'(z^{(i)})2(a^{(i)}-y) = a^{(i)}\frac{\partial C}{\partial b^{(i)}}$$

The other partial derivatives are computed similarly.

## Testing on Digit Recognition






