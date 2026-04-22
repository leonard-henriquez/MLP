#ifndef MLP_TESTS_CHECK_H
#define MLP_TESTS_CHECK_H

// Minimal test helpers: each failed check is reported and counted, and the
// test program exits with a non zero status if anything failed.

#include <cmath>
#include <iostream>

namespace check
{
extern int failures;
}

#define CHECK(condition)                                                      \
    do                                                                        \
    {                                                                         \
        if (!(condition))                                                     \
        {                                                                     \
            std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK(" #condition \
                      << ") failed\n";                                        \
            check::failures++;                                                \
        }                                                                     \
    } while (0)

#define CHECK_CLOSE(actual, expected, tolerance)                                \
    do                                                                          \
    {                                                                           \
        double a_ = (actual), e_ = (expected);                                  \
        if (std::fabs(a_ - e_) > (tolerance))                                   \
        {                                                                       \
            std::cerr << __FILE__ << ':' << __LINE__ << ": " #actual " is " << a_ \
                      << ", expected " << e_ << '\n';                           \
            check::failures++;                                                  \
        }                                                                       \
    } while (0)

#define CHECK_THROWS(statement)                                               \
    do                                                                        \
    {                                                                         \
        bool thrown_ = false;                                                 \
        try                                                                   \
        {                                                                     \
            statement;                                                        \
        }                                                                     \
        catch (...)                                                           \
        {                                                                     \
            thrown_ = true;                                                   \
        }                                                                     \
        if (!thrown_)                                                         \
        {                                                                     \
            std::cerr << __FILE__ << ':' << __LINE__ << ": " #statement       \
                      << " did not throw\n";                                  \
            check::failures++;                                                \
        }                                                                     \
    } while (0)

#endif // MLP_TESTS_CHECK_H
