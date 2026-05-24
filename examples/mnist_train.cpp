// Train a network to recognise the handwritten digits of MNIST.
//
//     mnist_train [options]
//
//     --data DIR       directory of the MNIST files (default data/mnist)
//     --hidden N       neurons in the hidden layer (default 100)
//     --epochs N       passes over the 60000 training images (default 10)
//     --rate R         learning rate (default 0.1)
//     --limit N        only use the first N training images
//     --seed N         seed of the initial weights and of the shuffling
//     --output FILE    where to save the network (default data/mnist/mnist.mlp)
//
// After each epoch the network is evaluated on the 10000 test images, which
// it never sees during training, and saved.

#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlp/Mnist.h"
#include "mlp/Network.h"
#include "mlp/Random.h"

namespace
{

struct Options
{
    std::string data;
    std::size_t hidden;
    unsigned epochs;
    double rate;
    std::size_t limit;
    unsigned long seed;
    std::string output;
};

Options parse(int argc, char* argv[])
{
    Options options;
    options.data = "data/mnist";
    options.hidden = 100;
    options.epochs = 10;
    options.rate = 0.1;
    options.limit = 0;
    options.seed = 1;
    options.output = "data/mnist/mnist.mlp";

    for (int i = 1; i < argc; i++)
    {
        std::string name = argv[i];
        if (i + 1 >= argc)
            throw std::invalid_argument("missing value after " + name);
        std::string value = argv[++i];

        if (name == "--data")
            options.data = value;
        else if (name == "--hidden")
            options.hidden = static_cast<std::size_t>(std::atol(value.c_str()));
        else if (name == "--epochs")
            options.epochs = static_cast<unsigned>(std::atol(value.c_str()));
        else if (name == "--rate")
            options.rate = std::atof(value.c_str());
        else if (name == "--limit")
            options.limit = static_cast<std::size_t>(std::atol(value.c_str()));
        else if (name == "--seed")
            options.seed = static_cast<unsigned long>(std::atol(value.c_str()));
        else if (name == "--output")
            options.output = value;
        else
            throw std::invalid_argument("unknown option " + name);
    }
    if (options.hidden == 0 || options.rate <= 0)
        throw std::invalid_argument("--hidden and --rate must be positive");
    return options;
}

// Percentage of test images classified correctly, and the confusion matrix:
// confusion[expected][predicted].
double evaluate(const mlp::Network& net, const mlp::Mnist& test,
                std::vector<std::vector<unsigned> >& confusion)
{
    confusion.assign(10, std::vector<unsigned>(10, 0));
    std::vector<double> inputs;
    unsigned correct = 0;
    for (std::size_t i = 0; i < test.size(); i++)
    {
        test.inputs(i, inputs);
        std::size_t predicted = net.classify(inputs);
        confusion[test.label(i)][predicted]++;
        if (predicted == test.label(i))
            correct++;
    }
    return 100.0 * correct / test.size();
}

void printConfusion(const std::vector<std::vector<unsigned> >& confusion)
{
    std::cout << "Confusion matrix on the test set (rows: digit, columns: prediction)\n\n      ";
    for (int p = 0; p < 10; p++)
        std::cout << std::setw(6) << p;
    std::cout << "    accuracy\n";
    for (int d = 0; d < 10; d++)
    {
        unsigned total = 0;
        std::cout << std::setw(6) << d;
        for (int p = 0; p < 10; p++)
        {
            std::cout << std::setw(6) << confusion[d][p];
            total += confusion[d][p];
        }
        std::cout << std::setw(11) << std::fixed << std::setprecision(1)
                  << 100.0 * confusion[d][d] / (total ? total : 1) << "%\n";
    }
}

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        Options options = parse(argc, argv);

        std::cout << "Loading MNIST from " << options.data << "...\n";
        mlp::Mnist train = mlp::Mnist::loadDirectory(options.data, false);
        mlp::Mnist test = mlp::Mnist::loadDirectory(options.data, true);
        std::size_t trainSize = options.limit && options.limit < train.size() ? options.limit
                                                                                : train.size();
        std::cout << trainSize << " training images, " << test.size() << " test images of "
                  << train.rows() << 'x' << train.columns() << " pixels\n";

        mlp::Network net(train.pixelCount(), options.seed);
        net.addLayer(options.hidden).addLayer(10);
        std::cout << "Network " << train.pixelCount() << '-' << options.hidden << "-10, learning rate "
                  << options.rate << "\n\n";

        std::vector<std::vector<unsigned> > confusion;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Before training: test accuracy " << evaluate(net, test, confusion) << "%\n";

        mlp::Random random(options.seed);
        std::vector<std::size_t> order(trainSize);
        for (std::size_t i = 0; i < order.size(); i++)
            order[i] = i;
        std::vector<double> inputs, targets;

        for (unsigned epoch = 1; epoch <= options.epochs; epoch++)
        {
            for (std::size_t i = order.size(); i > 1; i--)
                std::swap(order[i - 1], order[random.index(i)]);

            double squaredError = 0.0;
            for (std::size_t n = 0; n < order.size(); n++)
            {
                train.inputs(order[n], inputs);
                train.targets(order[n], targets);
                squaredError += net.trainExample(inputs, targets, options.rate);
            }

            double accuracy = evaluate(net, test, confusion);
            std::cout << "Epoch " << std::setw(2) << epoch << ": training error "
                      << std::setprecision(4) << squaredError / (order.size() * 10.0)
                      << std::setprecision(2) << ", test accuracy " << accuracy << "%" << std::endl;
            net.save(options.output);
        }

        std::cout << "\nNetwork saved to " << options.output << "\n\n";
        printConfusion(confusion);
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cerr << "error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
