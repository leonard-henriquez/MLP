#include "Session.h"

#include <fstream>
#include <stdexcept>

namespace tui
{

const char* const Session::MNIST_DIRECTORY = "data/mnist";
const char* const Session::NETWORK_PATH = "data/mnist/mnist.mlp";

Session::Session() : m_training(0), m_test(0), m_network(0), m_triedFile(false)
{
}

Session::~Session()
{
    delete m_training;
    delete m_test;
    delete m_network;
}

bool Session::hasMnist() const
{
    const char* files[] = {"train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                           "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"};
    for (int i = 0; i < 4; i++)
    {
        std::ifstream file((std::string(MNIST_DIRECTORY) + "/" + files[i]).c_str());
        if (!file)
            return false;
    }
    return true;
}

const mlp::Mnist& Session::training()
{
    if (!m_training)
        m_training = new mlp::Mnist(mlp::Mnist::loadDirectory(MNIST_DIRECTORY, false));
    return *m_training;
}

const mlp::Mnist& Session::test()
{
    if (!m_test)
        m_test = new mlp::Mnist(mlp::Mnist::loadDirectory(MNIST_DIRECTORY, true));
    return *m_test;
}

const mlp::Network* Session::network()
{
    if (!m_network && !m_triedFile)
    {
        std::string error;
        if (!loadNetwork(error))
            m_origin = error;
    }
    return m_network;
}

void Session::setNetwork(const mlp::Network& network, const std::string& origin)
{
    mlp::Network* copy = new mlp::Network(network);
    delete m_network;
    m_network = copy;
    m_origin = origin;
}

bool Session::loadNetwork(std::string& error)
{
    m_triedFile = true;
    try
    {
        mlp::Network network = mlp::Network::load(NETWORK_PATH);
        if (network.inputCount() != 28 * 28 || network.outputCount() != 10)
            throw std::runtime_error(std::string(NETWORK_PATH) + " is not a network for MNIST");
        setNetwork(network, NETWORK_PATH);
        return true;
    }
    catch (const std::exception& e)
    {
        error = e.what();
        return false;
    }
}

} // namespace tui
