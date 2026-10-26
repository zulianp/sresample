#ifndef SRESAMPLE_TEST_HPP
#define SRESAMPLE_TEST_HPP

#include <math.h>
#include <stdio.h>

// The tests' assertions: on failure they print where and return 1 from the test function.

#define SRESAMPLE_TEST_ASSERT(expr)                                                       \
    do {                                                                                  \
        if (!(expr)) {                                                                    \
            fprintf(stderr, "%s:%d: assertion failed: %s\n", __FILE__, __LINE__, #expr); \
            return 1;                                                                     \
        }                                                                                 \
    } while (0)

#define SRESAMPLE_TEST_APPROXEQ(a, b, tol)                                                                          \
    do {                                                                                                            \
        const double a__ = (double)(a), b__ = (double)(b);                                                          \
        if (!(fabs(a__ - b__) <= (tol))) {                                                                          \
            fprintf(stderr, "%s:%d: %s = %.17g, %s = %.17g, tolerance %g\n", __FILE__, __LINE__, #a, a__, #b, b__, \
                    (double)(tol));                                                                                 \
            return 1;                                                                                               \
        }                                                                                                           \
    } while (0)

#define SRESAMPLE_TEST_EQ(a, b) SRESAMPLE_TEST_ASSERT((a) == (b))

#define SRESAMPLE_RUN_TEST(test)                         \
    do {                                                 \
        const int err__ = test();                        \
        printf("%s: %s\n", #test, err__ ? "FAILED" : "ok"); \
        err += err__;                                    \
    } while (0)

#endif  // SRESAMPLE_TEST_HPP
