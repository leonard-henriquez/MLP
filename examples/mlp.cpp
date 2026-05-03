// Train a multilayer perceptron with one hidden layer on a dataset.
//
//     mlp examples/data/xor2.txt [hidden-neurons] [learning-rate] [max-epochs]

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "mlp/Dataset.h"
#include "mlp/Network.h"

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "usage: " << argv[0]
                  << " <data-file> [hidden-neurons] [learning-rate] [max-epochs]\n";
        return EXIT_FAILURE;
    }

    try
    {
        mlp::Dataset data = mlp::Dataset::load(argv[1]);
        int hidden = argc > 2 ? std::atoi(argv[2]) : 3;
        if (hidden <= 0)
            throw std::invalid_argument("the number of hidden neurons must be positive");

        mlp::TrainingOptions options;
        if (argc > 3)
            options.learningRate = std::atof(argv[3]);
        if (argc > 4)
            options.maxEpochs = static_cast<unsigned>(std::atoi(argv[4]));

        mlp::Network net(data.inputCount());
        net.addLayer(hidden).addLayer(data.outputCount());

        std::cout << "Network " << data.inputCount() << '-' << hidden << '-'
                  << data.outputCount() << ", learning rate " << options.learningRate << "\n\n";

        // Train in slices to report the progress.
        const unsigned slices = 10;
        unsigned maxEpochs = options.maxEpochs, epochs = 0;
        mlp::TrainingResult result;
        result.converged = false;
        for (unsigned s = 0; s < slices && !result.converged; s++)
        {
            options.maxEpochs = maxEpochs * (s + 1) / slices - epochs;
            result = net.train(data, options);
            epochs += result.epochs;
            std::cout << "epoch " << epochs << ": error " << result.error << '\n';
        }

        std::cout << '\n' << (result.converged ? "Learned" : "Not learned") << " in " << epochs
                  << " epochs\n\n";

        for (std::size_t i = 0; i < data.size(); i++)
        {
            std::vector<double> outputs = net.predict(data[i].inputs);
            for (std::size_t j = 0; j < data[i].inputs.size(); j++)
                std::cout << data[i].inputs[j] << ' ';
            std::cout << "->";
            for (std::size_t k = 0; k < outputs.size(); k++)
                std::cout << ' ' << outputs[k] << " (expected " << data[i].targets[k] << ')';
            std::cout << '\n';
        }
        return result.converged ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception& e)
    {
        std::cerr << "error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
