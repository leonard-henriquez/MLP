#ifndef MLP_ACTIVATION_H
#define MLP_ACTIVATION_H

namespace mlp
{

enum ActivationType
{
    STEP,    // 1 if x > 0, 0 otherwise
    SIGMOID  // 1 / (1 + exp(-k.x))
};

// Activation function of a neuron, applied to the weighted sum of its inputs.
class Activation
{
public:
    explicit Activation(ActivationType type = SIGMOID, double steepness = 1.0);

    double value(double x) const;
    // Derivative with respect to x. The step function is not differentiable,
    // so its derivative is taken as 1 (this gives the perceptron learning rule).
    double derivative(double x) const;

    ActivationType type() const { return m_type; }
    double steepness() const { return m_steepness; }

private:
    ActivationType m_type;
    double m_steepness;
};

} // namespace mlp

#endif // MLP_ACTIVATION_H
