#include "mlp/Network.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace mlp
{

namespace
{

const char* activationName(ActivationType type)
{
    switch (type)
    {
    case STEP:
        return "step";
    case SIGMOID:
        return "sigmoid";
    case TANH:
        return "tanh";
    case LINEAR:
        return "linear";
    }
    return "?";
}

ActivationType activationType(const std::string& name)
{
    if (name == "step")
        return STEP;
    if (name == "sigmoid")
        return SIGMOID;
    if (name == "tanh")
        return TANH;
    if (name == "linear")
        return LINEAR;
    throw std::runtime_error("unknown activation '" + name + "'");
}

void expect(std::istream& in, const std::string& keyword)
{
    std::string word;
    if (!(in >> word) || word != keyword)
        throw std::runtime_error("expected '" + keyword + "' in network file");
}

template <typename T>
T read(std::istream& in)
{
    T value;
    if (!(in >> value))
        throw std::runtime_error("unexpected value in network file");
    return value;
}

} // namespace

TrainingOptions::TrainingOptions()
    : learningRate(0.5), maxEpochs(10000), targetError(0.001), shuffle(true)
{
}

Network::Network(std::size_t inputCount, unsigned long seed)
    : m_inputCount(inputCount), m_random(seed)
{
    if (inputCount == 0)
        throw std::invalid_argument("a network needs at least one input");
}

Network& Network::addLayer(std::size_t neuronCount, const Activation& activation)
{
    std::size_t inputs = m_layers.empty() ? m_inputCount : m_layers.back().size();
    Layer layer(inputs, neuronCount, activation);
    layer.randomize(m_random, 1.0 / std::sqrt(static_cast<double>(inputs)));
    m_layers.push_back(layer);
    return *this;
}

std::size_t Network::outputCount() const
{
    return m_layers.empty() ? 0 : m_layers.back().size();
}

std::vector<double> Network::predict(const std::vector<double>& inputs) const
{
    if (m_layers.empty())
        throw std::logic_error("the network has no layer");

    std::vector<double> values = inputs, next;
    for (std::size_t k = 0; k < m_layers.size(); k++)
    {
        m_layers[k].compute(values, next);
        values.swap(next);
    }
    return values;
}

std::size_t Network::classify(const std::vector<double>& inputs) const
{
    std::vector<double> outputs = predict(inputs);
    return static_cast<std::size_t>(std::max_element(outputs.begin(), outputs.end()) - outputs.begin());
}

double Network::trainExample(const std::vector<double>& inputs,
                             const std::vector<double>& targets, double learningRate)
{
    if (m_layers.empty())
        throw std::logic_error("the network has no layer");
    if (targets.size() != outputCount())
        throw std::invalid_argument("wrong number of targets");

    const std::vector<double>* values = &inputs;
    for (std::size_t k = 0; k < m_layers.size(); k++)
        values = &m_layers[k].forward(*values);
    const std::vector<double>& outputs = *values;

    // E = 1/2 sum_k (output_k - target_k)^2, so dE/doutput_k = output_k - target_k.
    double squaredError = 0.0;
    std::vector<double> gradient(outputs.size()), previous;
    for (std::size_t k = 0; k < outputs.size(); k++)
    {
        gradient[k] = outputs[k] - targets[k];
        squaredError += gradient[k] * gradient[k];
    }

    for (std::size_t k = m_layers.size(); k-- > 0;)
    {
        m_layers[k].backward(gradient, learningRate, previous);
        gradient.swap(previous);
    }
    return squaredError;
}

TrainingResult Network::train(const Dataset& data, const TrainingOptions& options)
{
    checkDataset(data);

    std::vector<std::size_t> order(data.size());
    for (std::size_t i = 0; i < order.size(); i++)
        order[i] = i;

    TrainingResult result;
    result.epochs = 0;
    result.error = error(data);
    result.converged = result.error <= options.targetError;

    while (!result.converged && result.epochs < options.maxEpochs)
    {
        if (options.shuffle)
        {
            // Fisher-Yates shuffle
            for (std::size_t i = order.size(); i > 1; i--)
                std::swap(order[i - 1], order[m_random.index(i)]);
        }

        for (std::size_t i = 0; i < order.size(); i++)
        {
            const Example& example = data[order[i]];
            trainExample(example.inputs, example.targets, options.learningRate);
        }

        result.epochs++;
        result.error = error(data);
        result.converged = result.error <= options.targetError;
    }
    return result;
}

double Network::error(const Dataset& data) const
{
    checkDataset(data);
    if (data.empty())
        return 0.0;

    double sum = 0.0;
    for (std::size_t n = 0; n < data.size(); n++)
    {
        std::vector<double> outputs = predict(data[n].inputs);
        for (std::size_t k = 0; k < outputs.size(); k++)
        {
            double difference = data[n].targets[k] - outputs[k];
            sum += difference * difference;
        }
    }
    return sum / (data.size() * outputCount());
}

void Network::checkDataset(const Dataset& data) const
{
    if (m_layers.empty())
        throw std::logic_error("the network has no layer");
    if (data.inputCount() != inputCount() || data.outputCount() != outputCount())
        throw std::invalid_argument("dataset does not match the network");
}

void Network::save(std::ostream& out) const
{
    std::streamsize precision = out.precision(std::numeric_limits<double>::digits10 + 2);

    out << "mlp 1\n";
    out << "inputs " << m_inputCount << '\n';
    out << "layers " << m_layers.size() << '\n';
    for (std::size_t k = 0; k < m_layers.size(); k++)
    {
        const Layer& layer = m_layers[k];
        out << "layer " << layer.size() << ' ' << activationName(layer.activation().type()) << ' '
            << layer.activation().steepness() << '\n';
        // One line per neuron: its weights, then its bias.
        for (std::size_t j = 0; j < layer.size(); j++)
        {
            for (std::size_t i = 0; i < layer.inputCount(); i++)
                out << layer.weight(j, i) << ' ';
            out << layer.bias(j) << '\n';
        }
    }
    out.precision(precision);
}

void Network::save(const std::string& path) const
{
    std::ofstream file(path.c_str());
    if (!file)
        throw std::runtime_error("cannot write network file '" + path + "'");
    save(file);
    if (!file)
        throw std::runtime_error("error while writing network file '" + path + "'");
}

Network Network::load(std::istream& in)
{
    expect(in, "mlp");
    if (read<int>(in) != 1)
        throw std::runtime_error("unsupported network file version");

    expect(in, "inputs");
    long inputCount = read<long>(in);
    expect(in, "layers");
    long layerCount = read<long>(in);
    if (inputCount <= 0 || layerCount < 0)
        throw std::runtime_error("invalid network dimensions");

    Network net(static_cast<std::size_t>(inputCount));
    for (long k = 0; k < layerCount; k++)
    {
        expect(in, "layer");
        long size = read<long>(in);
        if (size <= 0)
            throw std::runtime_error("invalid layer size");
        ActivationType type = activationType(read<std::string>(in));
        double steepness = read<double>(in);

        net.addLayer(static_cast<std::size_t>(size), Activation(type, steepness));
        Layer& layer = net.m_layers.back();
        for (std::size_t j = 0; j < layer.size(); j++)
        {
            for (std::size_t i = 0; i < layer.inputCount(); i++)
                layer.setWeight(j, i, read<double>(in));
            layer.setBias(j, read<double>(in));
        }
    }
    return net;
}

Network Network::load(const std::string& path)
{
    std::ifstream file(path.c_str());
    if (!file)
        throw std::runtime_error("cannot open network file '" + path + "'");
    return load(file);
}

} // namespace mlp
