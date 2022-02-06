
SHELL := /bin/bash
FLAGS = -DHAVE_CBLAS=1 -llapack -lblas
AUTOD_SRCS = ../tensor_autograd.cpp ../tensor_autograd.hpp

.PHONY: clean env optional

all: 1_ex.out 2_ex.out 3_ex.out 4_ex.out linear_regression_1.out linear_regression_2.out

1_ex.out : 1_example.cpp autodiff.o
	g++ $^ -o $@

2_ex.out : 2_example.cpp autodiff.o
	g++ $^ -o $@

3_ex.out : 3_example.cpp autodiff.o
	g++ $^ $(FLAGS) -o $@

4_ex.out : 4_example.cpp autodiff.o
	g++ $^ $(FLAGS) -o $@

linear_regression_1.out : linear_regression_1.cpp autodiff.o
	g++ $^ $(FLAGS) -o $@

linear_regression_2.out : linear_regression_2.cpp autodiff.o
	g++ $^ $(FLAGS) -o $@

autodiff.o : ../tensor_autograd.cpp
	g++ -g -c $^ $(FLAGS) -o $@

clean:
	rm -f *.out *.o
	clear
