TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle qt

SOURCES += main.cpp \
    multilayerperceptron.cpp

INCLUDEPATH += $PWD/Eigen

include(deployment.pri)
qtcAddDeployment()

HEADERS += \
    mlpmath.h \
    types.h \
    multilayerperceptron.h \
    multilayerperceptron_global.h

CONFIG(debug, debug|release) {
    message(Debug build!)
} else {
    INCLUDEPATH += /opt/intel/mkl/include
    LIBS += -L/opt/intel/mkl/lib/intel64 \
        -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
        -L/opt/intel/lib/intel64 \
        -liomp5 -lpthread -lm
    DEFINES += NDEBUG
    DEFINES += EIGEN_USE_MKL_ALL
    message(Release build!)
}

QMAKE_CXXFLAGS_RELEASE -= -O2
QMAKE_CXXFLAGS_RELEASE += -fast -march=core2 -openmp -static
