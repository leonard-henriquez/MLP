#include "mlp/Dataset.h"

#include <fstream>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace mlp
{

namespace
{

std::string trim(const std::string& s)
{
    const char* blanks = " \t\r\n";
    std::string::size_type begin = s.find_first_not_of(blanks);
    if (begin == std::string::npos)
        return "";
    std::string::size_type end = s.find_last_not_of(blanks);
    return s.substr(begin, end - begin + 1);
}

double toNumber(const std::string& s)
{
    std::istringstream stream(trim(s));
    double value;
    if (!(stream >> value) || !stream.eof())
        throw std::runtime_error("invalid number: '" + trim(s) + "'");
    return value;
}

// Read a header entry such as "exemples 4".
std::size_t readHeader(const std::string& entry, const std::string& key)
{
    std::istringstream stream(entry);
    std::string name;
    long value;
    if (!(stream >> name >> value) || name != key || value < 0)
        throw std::runtime_error("expected '" + key + " <n>' in header, got '" + trim(entry) + "'");
    return static_cast<std::size_t>(value);
}

} // namespace

Dataset::Dataset() : m_inputCount(0), m_outputCount(0)
{
}

Dataset::Dataset(std::size_t inputCount, std::size_t outputCount)
    : m_inputCount(inputCount), m_outputCount(outputCount)
{
}

Dataset Dataset::load(const std::string& path)
{
    std::ifstream file(path.c_str());
    if (!file)
        throw std::runtime_error("cannot open data file '" + path + "'");
    return parse(file);
}

Dataset Dataset::parse(std::istream& in)
{
    std::vector<std::string> entries;
    std::string entry;
    while (std::getline(in, entry, ';'))
    {
        entry = trim(entry);
        if (!entry.empty())
            entries.push_back(entry);
    }

    if (entries.size() < 2)
        throw std::runtime_error("missing header");

    std::size_t exampleCount = readHeader(entries[0], "exemples");
    std::size_t inputCount = readHeader(entries[1], "entrees");
    std::size_t outputCount = 1;
    std::size_t first = 2;
    if (entries.size() > 2 && entries[2].compare(0, 7, "sorties") == 0)
    {
        outputCount = readHeader(entries[2], "sorties");
        first = 3;
    }

    if (entries.size() - first != exampleCount)
    {
        std::ostringstream message;
        message << "header announces " << exampleCount << " examples, found "
                << entries.size() - first;
        throw std::runtime_error(message.str());
    }

    Dataset data(inputCount, outputCount);
    for (std::size_t i = first; i < entries.size(); i++)
    {
        std::vector<double> values;
        std::istringstream row(entries[i]);
        std::string field;
        while (std::getline(row, field, '/'))
            values.push_back(toNumber(field));

        if (values.size() != inputCount + outputCount)
            throw std::runtime_error("wrong number of values in example '" + entries[i] + "'");

        std::vector<double> inputs(values.begin(), values.begin() + inputCount);
        std::vector<double> targets(values.begin() + inputCount, values.end());
        data.add(inputs, targets);
    }
    return data;
}

void Dataset::add(const std::vector<double>& inputs, const std::vector<double>& targets)
{
    if (inputs.size() != m_inputCount || targets.size() != m_outputCount)
        throw std::invalid_argument("example does not match the dataset dimensions");

    Example example;
    example.inputs = inputs;
    example.targets = targets;
    m_examples.push_back(example);
}

std::ostream& operator<<(std::ostream& out, const Dataset& data)
{
    for (std::size_t i = 0; i < data.size(); i++)
    {
        for (std::size_t j = 0; j < data[i].inputs.size(); j++)
            out << data[i].inputs[j] << ' ';
        out << "->";
        for (std::size_t j = 0; j < data[i].targets.size(); j++)
            out << ' ' << data[i].targets[j];
        out << '\n';
    }
    return out;
}

} // namespace mlp
