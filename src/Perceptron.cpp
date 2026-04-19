#include "mlp/Perceptron.h"

#include <stdexcept>

#include "mlp/Activation.h"

namespace mlp
{

Perceptron::Perceptron(std::size_t inputCount, double learningRate)
    : m_weights(inputCount, 1.0), m_threshold(1.0), m_learningRate(learningRate), m_epochs(0)
{
}

double Perceptron::output(const std::vector<double>& inputs) const
{
    if (inputs.size() != m_weights.size())
        throw std::invalid_argument("wrong number of inputs");

    double sum = 0.0;
    for (std::size_t i = 0; i < inputs.size(); i++)
        sum += m_weights[i] * inputs[i];
    return Activation(STEP).value(sum - m_threshold);
}

bool Perceptron::train(const Dataset& data, unsigned maxEpochs)
{
    if (data.inputCount() != inputCount() || data.outputCount() != 1)
        throw std::invalid_argument("dataset does not match the perceptron");

    for (m_epochs = 1; m_epochs <= maxEpochs; m_epochs++)
    {
        bool allCorrect = true;
        for (std::size_t n = 0; n < data.size(); n++)
        {
            const Example& example = data[n];
            double error = example.targets[0] - output(example.inputs);
            if (error == 0.0)
                continue;

            allCorrect = false;
            for (std::size_t i = 0; i < m_weights.size(); i++)
                m_weights[i] += m_learningRate * error * example.inputs[i];
            m_threshold -= m_learningRate * error;
        }
        if (allCorrect)
            return true;
    }
    m_epochs = maxEpochs;
    return false;
}

std::ostream& operator<<(std::ostream& out, const Perceptron& perceptron)
{
    for (std::size_t i = 0; i < perceptron.inputCount(); i++)
        out << "weight " << i << ": " << perceptron.weights()[i] << '\n';
    out << "threshold: " << perceptron.threshold() << '\n';
    return out;
}

} // namespace mlp
