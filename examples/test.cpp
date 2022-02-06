#include<xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <iostream>

int main(){
	xt::xarray<double> arr1 {{1,2},{1,2}};
	xt::xarray<double> arr2 {{1,2},{1,2}};
	std::cout << xt::linalg::dot(arr1, arr2);
}
