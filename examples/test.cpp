#include<xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xview.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xrandom.hpp>
#include <fstream>
#include <iostream>
#include "../tools/train_test_split.cpp"

using lodo = long double;
using namespace std;

int main(){


	ifstream file ("../data/data1");
	auto d = xt::load_csv<lodo>(file);
	auto data = xt::view(d, xt::range(0,12), xt::all());

	size_t M = data.shape(0), N = data.shape(1);

	xt::xarray<lodo> X, y;
	X = xt::view(data, xt::all(), xt::range(0, N-1));
	y = xt::col(data, N-1);


	//printf("%d %d %d\n", X.shape(0), X.shape(1), y.shape(0) );
	
	auto spd = train_test_split(X, y, 0.25, 69);
	auto X_train = spd.X_train;
	auto X_test = spd.X_test;
	auto y_train = spd.y_train;
	auto y_test = spd.y_test;

	cout << data << endl;
	cout << X_train.shape(0) << endl << X_test.shape(0) << endl;
	cout << y_test << y_train;

	/*
	xt::xarray<double> arr1 {{1,2,3},{4,5,6},{7,8,9}};
	arr1 = xt::row(arr1, 1)[2];
	std::cout << arr1;
	*/
}
