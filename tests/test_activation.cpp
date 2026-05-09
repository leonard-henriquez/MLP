#include <cstddef>

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

    Activation tanh(mlp::TANH);
    CHECK_CLOSE(tanh.value(0.0), 0.0, 1e-12);
    CHECK_CLOSE(tanh.value(1.0), 0.761594, 1e-6);
    CHECK_CLOSE(tanh.value(-1.0), -0.761594, 1e-6);

    Activation linear(mlp::LINEAR, 2.0);
    CHECK_CLOSE(linear.value(-1.5), -3.0, 1e-12);

    // Compare derivatives with finite differences.
    const Activation differentiable[] = {
        sigmoid, Activation(mlp::SIGMOID, 3.0), tanh, Activation(mlp::TANH, 0.5), linear
    };
    const double h = 1e-6;
    for (std::size_t n = 0; n < sizeof differentiable / sizeof differentiable[0]; n++)
    {
        const Activation& f = differentiable[n];
        for (double x = -3.0; x <= 3.0; x += 0.5)
        {
            double numeric = (f.value(x + h) - f.value(x - h)) / (2 * h);
            CHECK_CLOSE(f.derivative(x), numeric, 1e-6);
        }
    }
}
