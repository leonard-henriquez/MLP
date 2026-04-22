#include <sstream>

#include "check.h"
#include "mlp/Dataset.h"

using mlp::Dataset;

namespace
{

Dataset parse(const char* text)
{
    std::istringstream in(text);
    return Dataset::parse(in);
}

} // namespace

void testDataset()
{
    Dataset data = parse("exemples 4; entrees 2;\n"
                         "1/1/1;\n"
                         "0/1/1;\n"
                         "1/0/1;\n"
                         "0/0/0;\n");
    CHECK(data.size() == 4);
    CHECK(data.inputCount() == 2);
    CHECK(data.outputCount() == 1);
    CHECK(data[1].inputs[0] == 0.0);
    CHECK(data[1].inputs[1] == 1.0);
    CHECK(data[3].targets[0] == 0.0);

    // Windows line endings, spaces and decimal values.
    Dataset decimals = parse("exemples 1; entrees 1;\r\n 0.25 / -1.5 ;\r\n");
    CHECK(decimals.size() == 1);
    CHECK(decimals[0].inputs[0] == 0.25);
    CHECK(decimals[0].targets[0] == -1.5);

    Dataset outputs = parse("exemples 1; entrees 2; sorties 2; 1/0/0/1;");
    CHECK(outputs.outputCount() == 2);
    CHECK(outputs[0].targets[1] == 1.0);

    CHECK_THROWS(parse(""));
    CHECK_THROWS(parse("examples 1; entrees 1; 0/0;"));
    CHECK_THROWS(parse("exemples 2; entrees 1; 0/0;"));
    CHECK_THROWS(parse("exemples 1; entrees 2; 0/0;"));
    CHECK_THROWS(parse("exemples 1; entrees 1; 0/x;"));
    CHECK_THROWS(Dataset::load("does/not/exist.txt"));

    CHECK(Dataset::load("data/OU2.txt").size() == 4);
    CHECK(Dataset::load("data/ET3.txt").inputCount() == 3);
}
