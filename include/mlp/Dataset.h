#ifndef MLP_DATASET_H
#define MLP_DATASET_H

#include <cstddef>
#include <istream>
#include <string>
#include <vector>

namespace mlp
{

struct Example
{
    std::vector<double> inputs;
    std::vector<double> targets;
};

// A set of training examples.
//
// Text format (whitespace is ignored, entries are separated by ';'):
//
//     exemples 4; entrees 2;
//     1/1/1;
//     0/1/1;
//     1/0/1;
//     0/0/0;
//
// The header gives the number of examples and the number of inputs. Each
// example lists its inputs followed by the expected outputs, separated by
// '/'. An optional "sorties N;" entry in the header sets the number of
// outputs (1 by default).
class Dataset
{
public:
    Dataset();
    Dataset(std::size_t inputCount, std::size_t outputCount);

    // Throw std::runtime_error if the file cannot be read or is malformed.
    static Dataset load(const std::string& path);
    static Dataset parse(std::istream& in);

    void add(const std::vector<double>& inputs, const std::vector<double>& targets);

    std::size_t size() const { return m_examples.size(); }
    bool empty() const { return m_examples.empty(); }
    std::size_t inputCount() const { return m_inputCount; }
    std::size_t outputCount() const { return m_outputCount; }

    const Example& operator[](std::size_t i) const { return m_examples[i]; }

private:
    std::size_t m_inputCount;
    std::size_t m_outputCount;
    std::vector<Example> m_examples;
};

std::ostream& operator<<(std::ostream& out, const Dataset& data);

} // namespace mlp

#endif // MLP_DATASET_H
