#include <xtensor/xtensor.hpp>

template<typename T>
xt::xarray<T> rmse(const xt::xarray<T> &y_test, const xt::xarray<T> &y_pred){
	xt::xarray<T> squared_diffs = xt::pow((y_test - y_pred), 2);
	size_t N = y_test.size();
	return xt::sum(squared_diffs, {1}) / N;
}
