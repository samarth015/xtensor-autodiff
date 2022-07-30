#include "../xtad/tensor_autograd.cpp"
#include "../tools/train_test_split.cpp"
#include <xtensor/xcsv.hpp>
#include <fstream>
#include <sstream>

using lodo = long double;
using namespace std;
using tensor = xtad::xarray<lodo>;

int main(int argc, char **argv){

	string filepath = (argc > 1 ? argv[1] : "../data/data1");
	ifstream file (filepath);
	auto data = xt::load_csv<lodo>(file);

	size_t M = data.shape(0), N = data.shape(1);
	cout << M << ' ' << N << endl;

	xt::xarray<lodo> X, y;
	X = xt::view(data, xt::all(), xt::range(0, N-1));
	y = xt::col(data, N-1);

	auto spd = train_test_split(X, y, 0.25, 69);
	auto X_train = spd.X_train, X_test = spd.X_test, y_train = spd.y_train, y_test = spd.y_test;


	double learning_rate { 0.05 };
	tensor K{0,0,0,0,0}, C{0};
	K.reshape({5,1});

	// printf("Slope shape %d %d\n", K.val().shape(0), K.val().shape(1));

	for(size_t epoch{1}; epoch <= 10; epoch++){
		for(size_t i = 0; i < X_train.shape(0); i++){
			//reading ith line
			lodo y_real = y_train[i];

			xt::xarray<lodo> row = xt::row(X_train, i);
			row.reshape({1,5}); // K is represented as a 2D matrix of 1 col so representing
								// X as a 2D matrix of 1 row
			tensor X (row);
			// printf("X shape %d %d\n", X.val().shape(0), X.val().shape(1));
			cout << X << endl;

			tensor Y = xtad::dot(X,K) + C;

			tensor Loss = xtad::pow(xtad::pow( Y - y_real, 2 ), 0.5);

			/*
			cout << Loss <<endl;
			cout << K << endl;
			cout << C << endl;
			*/

			Loss.backward();

			/*
			cout << Loss <<endl;
			cout << K << endl;
			cout << C << endl;
			*/

			K = K + K.grad() * -1 * learning_rate;
			C = C + C.grad() * -1 * learning_rate;

			tensor::clear_buffer();
		}
		cout << epoch << " : " << K.val() << "   " << C.val() << endl ;
	}

	cout << "\nPredicted Y vs Actual Y" << endl;
	tensor rmse {0};
	int test_count {0};

	for(size_t i = 0; i < X_test.shape(0); i++){
		test_count++;
			lodo y_real = y_test[i];

			xt::xarray<lodo> row = xt::row(X_test, i);
			tensor X (row);

			tensor Y = xtad::dot(X,K) + C;

		cout << Y.val() << ' ' << y_real << endl;
		rmse += xtad::pow(Y - y_real, 2);
	}
	rmse /= test_count;
	rmse = xtad::pow(rmse, 0.5);
	cout << "\nRoot Mean Square Error : " << rmse.val() << endl;
}

