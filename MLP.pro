TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle qt

SOURCES += main.cpp \
    multilayerperceptron.cpp

INCLUDEPATH += $PWD/Eigen

include(deployment.pri)
qtcAddDeployment()

HEADERS += multilayerperceptron.h \
    io.h \
    includes.h

INCLUDEPATH += /opt/intel/mkl/include
LIBS += -L/opt/intel/mkl/lib/intel64 \
        -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core \
        -L/opt/intel/lib/intel64 \
        -liomp5 -lpthread -lm
DEFINES += NDEBUG \
        EIGEN_USE_MKL_ALL

QMAKE_CXXFLAGS += -fast -march=corei7 -qopenmp -static
