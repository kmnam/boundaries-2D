# ------------------------------------------------------------ #
# Common Makefile for all scripts under src/.
#
# Authors:
#     Kee-Myoung Nam, Department of Systems Biology, Harvard Medical School
# Last updated:
#     4/11/2019
# ------------------------------------------------------------ #

CXX = g++-8

CXXFLAGS = --std=c++11 -frounding-math

INCLUDES = -I/usr/local/Cellar/cgal/4.14.1/include -I/usr/local/Cellar/boost/1.71.0/include -I/usr/local/Cellar/eigen/3.3.7/include/eigen3 -I/anaconda3/include -I/Users/kmnam/Dropbox/gene-regulation/projects/dual-numbers/include

LDFLAGS = -L/usr/local/Cellar/cgal/4.14.1/lib -L/usr/local/Cellar/boost/1.71.0/lib 

LIBS = -lCGAL -lgmp -lboost_unit_test_framework

SRC_DIR = src

SRC_FILES = src/testBoundaries.cpp src/testReverseSQP.cpp src/testDualNumberSQP.cpp src/testDoubleSQP.cpp src/testBoundaryFinder.cpp src/testLinearConstraints.cpp

BIN_DIR = bin

BIN_FILES = $(addprefix $(BIN_DIR)/, $(notdir $(patsubst %.cpp, %.out, $(SRC_FILES))))

all: $(BIN_FILES)

$(BIN_DIR)/%.out: $(SRC_DIR)/%.cpp 
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) $< -o $(@:.out=) $(LIBS)

clean: $(patsubst %.out, %.clean, $(BIN_FILES))

%.clean:
	$(RM) $(@:.clean=)
