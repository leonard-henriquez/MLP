#include <iostream>

#include "check.h"

namespace check
{
int failures = 0;
}

void testActivation();
void testDataset();
void testPerceptron();
void testRandom();
void testLayer();
void testNetwork();
void testBackpropagation();
void testTraining();

int main()
{
    testActivation();
    testDataset();
    testPerceptron();
    testRandom();
    testLayer();
    testNetwork();
    testBackpropagation();
    testTraining();

    if (check::failures > 0)
    {
        std::cerr << check::failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "All tests passed\n";
    return 0;
}
