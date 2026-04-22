#include "check.h"
#include "mlp/Activation.h"

using mlp::Activation;

void testActivation()
{
    Activation step(mlp::STEP);
    CHECK(step.value(-0.5) == 0.0);
    CHECK(step.value(0.0) == 0.0);
    CHECK(step.value(0.5) == 1.0);

    Activation sigmoid(mlp::SIGMOID);
    CHECK_CLOSE(sigmoid.value(0.0), 0.5, 1e-12);
    CHECK_CLOSE(sigmoid.value(2.0), 0.880797, 1e-6);
    CHECK_CLOSE(sigmoid.value(-2.0), 1.0 - 0.880797, 1e-6);

    // A steep sigmoid behaves like the step function.
    Activation steep(mlp::SIGMOID, 100.0);
    CHECK_CLOSE(steep.value(0.1), 1.0, 1e-4);
    CHECK_CLOSE(steep.value(-0.1), 0.0, 1e-4);

    // Compare derivatives with finite differences.
    const double h = 1e-6;
    for (double x = -3.0; x <= 3.0; x += 0.5)
    {
        double numeric = (sigmoid.value(x + h) - sigmoid.value(x - h)) / (2 * h);
        CHECK_CLOSE(sigmoid.derivative(x), numeric, 1e-6);
    }
}
