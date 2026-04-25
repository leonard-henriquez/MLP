#include "check.h"
#include "mlp/Random.h"

using mlp::Random;

void testRandom()
{
    // Reference value of the minimal standard generator after 10000 draws
    // starting from 1 (Park & Miller, with multiplier 48271).
    Random reference(0);
    double last = 0.0;
    for (int i = 0; i < 10000; i++)
        last = reference.uniform(0.0, 2147483646.0);
    CHECK_CLOSE(last, 399268537.0 - 1.0, 1e-3);

    Random a(7), b(7);
    for (int i = 0; i < 100; i++)
    {
        double x = a.uniform(-1.0, 1.0);
        CHECK(x >= -1.0 && x < 1.0);
        CHECK(x == b.uniform(-1.0, 1.0));
        CHECK(a.index(3) < 3);
        b.index(3);
    }
}
