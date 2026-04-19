#ifndef MLP_PERCEPTRON_H
#define MLP_PERCEPTRON_H

#include <cstddef>
#include <ostream>
#include <vector>

#include "mlp/Dataset.h"

namespace mlp
{

// Single neuron with a step activation, trained with the perceptron rule:
//
//     w_i <- w_i + rate * (expected - output) * x_i
//     threshold <- threshold - rate * (expected - output)
//
// It converges for any linearly separable problem (AND, OR, NOR...).
class Perceptron
{
public:
    explicit Perceptron(std::size_t inputCount, double learningRate = 0.1);

    double output(const std::vector<double>& inputs) const;

    // Run epochs until every example is classified correctly. Return false if
    // that did not happen within maxEpochs (the problem is probably not
    // linearly separable).
    bool train(const Dataset& data, unsigned maxEpochs = 1000);

    std::size_t inputCount() const { return m_weights.size(); }
    const std::vector<double>& weights() const { return m_weights; }
    double threshold() const { return m_threshold; }
    double learningRate() const { return m_learningRate; }
    unsigned epochs() const { return m_epochs; }

private:
    std::vector<double> m_weights;
    double m_threshold;
    double m_learningRate;
    unsigned m_epochs;
};

std::ostream& operator<<(std::ostream& out, const Perceptron& perceptron);

} // namespace mlp

#endif // MLP_PERCEPTRON_H
