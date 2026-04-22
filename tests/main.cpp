#include <iostream>

#include "check.h"

namespace check
{
int failures = 0;
}

void testActivation();
void testDataset();
void testPerceptron();

int main()
{
    testActivation();
    testDataset();
    testPerceptron();

    if (check::failures > 0)
    {
        std::cerr << check::failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "All tests passed\n";
    return 0;
}
