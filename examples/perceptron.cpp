// Train a single perceptron on a dataset and print what it learned.
//
//     perceptron data/OU2.txt [learning-rate]

#include <cstdlib>
#include <exception>
#include <iostream>

#include "mlp/Dataset.h"
#include "mlp/Perceptron.h"

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0] << " <data-file> [learning-rate]\n";
        return EXIT_FAILURE;
    }

    try
    {
        mlp::Dataset data = mlp::Dataset::load(argv[1]);
        double rate = argc > 2 ? std::atof(argv[2]) : 0.1;

        std::cout << "Examples:\n" << data << '\n';

        mlp::Perceptron perceptron(data.inputCount(), rate);
        bool learned = perceptron.train(data);

        if (learned)
            std::cout << "Learned in " << perceptron.epochs() << " epochs\n";
        else
            std::cout << "Not learned after " << perceptron.epochs()
                      << " epochs: the problem is probably not linearly separable\n";
        std::cout << perceptron << '\n';

        for (std::size_t i = 0; i < data.size(); i++)
        {
            for (std::size_t j = 0; j < data[i].inputs.size(); j++)
                std::cout << data[i].inputs[j] << ' ';
            std::cout << "-> " << perceptron.output(data[i].inputs)
                      << " (expected " << data[i].targets[0] << ")\n";
        }
        return learned ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception& e)
    {
        std::cerr << "error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
