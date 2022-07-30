
SHELL := /bin/bash
FLAGS = -DHAVE_CBLAS=1 -llapack -lblas


.PHONY: clean env optional


#test : test.cpp
#	g++ -g $^ $(FLAGS) -o $@

out.model1 : model_linear_regression_1.cpp
	g++ $^ $(FLAGS) -o $@

out.1_ex : 1_example.cpp
	g++ $^ $(FLAGS) -o $@

out.2_ex : 2_example.cpp
	g++ $^ $(FLAGS) -o $@

out.3_ex : 3_example.cpp
	g++ $^ $(FLAGS) -o $@

out.4_ex : 4_example.cpp
	g++ $^ $(FLAGS) -o $@


clean:
	rm -f out.*
