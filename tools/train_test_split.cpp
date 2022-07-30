#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <unordered_set>
#include <random>
#include <xtensor/xadapt.hpp>

template<typename T>
struct splitted_data{
	xt::xarray<T> X_train, X_test, y_train, y_test;
};

template<typename T>
splitted_data<T> train_test_split(xt::xarray<T> &X, xt::xarray<T> &y, double test_fraction = 0.25, double seed = 619){
	srand(seed);
	size_t M = X.shape(0);
	size_t test_size = test_fraction * M;

	std::vector<size_t> indx (M, 1);
	size_t rows_selected = 0;
	while(test_size != rows_selected){
		size_t i = rand() % M;
		if(indx[i]) 
			indx[i] = 0, rows_selected++;
	}

	std::vector<size_t> train_indx, test_indx;
	for(size_t i = 0; i < M; i++){
		if(indx[i]) train_indx.push_back(i);
		else test_indx.push_back(i);
	}

	auto xt_train_indx = xt::adapt(train_indx, {train_indx.size()});
	auto xt_test_indx = xt::adapt(test_indx, {test_size});

	splitted_data<T> spd{ xt::view(X, xt::keep(xt_train_indx), xt::all()),
						  xt::view(X, xt::keep(xt_test_indx), xt::all()),
						  xt::view(y, xt::keep(xt_train_indx), xt::all()),
						  xt::view(y, xt::keep(xt_test_indx), xt::all()) };

	return spd;
}
