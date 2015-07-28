#ifndef MULTILAYERPERCEPTRON_GLOBAL_H
#define MULTILAYERPERCEPTRON_GLOBAL_H

#include "types.h"
#include "mlpmath.h"

const realnumber LEARNING_RATE           = 0.5,
                 ALPHA_PLUS              = 0.01,
                 ALPHA_MINUS             = 0.04,
                 LAMBDA                  = 0.0002, // general
                 LAMBDA1                 = 0.00001, // last
                 LAMBDA2                 = 0.00001, // bias
                 MAX_ERROR               = 0.0001,
                 MAX_TIME                = 7,
                 MAX_DELTA_MQE           = 0.02,
                 LAST_LAYER_LINEAR       = 0,
                 ADAPTATIVELR            = 0;

#endif // MULTILAYERPERCEPTRON_GLOBAL_H
