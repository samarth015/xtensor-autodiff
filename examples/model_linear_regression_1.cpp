#include "../models/linear_regressor.cpp"
#include "../tools/train_test_split.cpp"
#include "../tools/root_mean_square_error.cpp"
#include <xtensor/xcsv.hpp>
#include <fstream>

using lodo = long double;
using namespace std;
using tensor = xtad::xarray<long double>;

int main(int argc, char **argv){

	string filepath = (argc > 1 ? argv[1] : "../data/data1");
	ifstream file (filepath);
	auto data = xt::load_csv<lodo>(file);

	size_t M = data.shape(0), N = data.shape(1);
	cout << M << ' ' << N << endl;

	xt::xarray<lodo> X, y;
	X = xt::view(data, xt::all(), xt::range(0, N-1));
	y = xt::col(data, N-1);

	auto spd = train_test_split(X, y, 0.25, 619);
	auto X_train = spd.X_train, X_test = spd.X_test, y_train = spd.y_train, y_test = spd.y_test;

	linear_regressor<lodo> model;
	model.fit(X_train, y_train, 0.001, 50);

	cout << model.biases() << endl;
	cout << model.coefficients() << endl;

	auto y_pred = model.predict(X_test);
	cout << y_pred << endl;

	y_test.reshape({1,(int)y_test.size()});
	cout  << y_test << endl;

	cout << "RMSE : " << rmse(y_test, y_pred) << endl;
}




