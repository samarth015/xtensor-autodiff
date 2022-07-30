#include "../xtad/tensor_autograd.cpp"
#include <fstream>
#include <sstream>

using lodo = long double;
using namespace std;
using tensor = xtad::xarray<lodo>;

int main(int argc, char **argv){

	string filepath = (argc > 1 ? argv[1] : "../data/data1");
	ifstream file (filepath);
	auto data = xt::load_csv<lodo>(file);

	int N = data.shape(0);   //number of samples

	cout << data.shape(0) << ' ' << data.shape(1) << endl;

	double learning_rate { 0.05 };
	tensor M{0,0,0,0,0}, C{0};
	M.reshape({5,1});

	printf("Slope shape %d %d\n", M.val().shape(0), M.val().shape(1));

	for(size_t epoch{1}; epoch <= 10; epoch++){
		for(size_t i = 0; i < N * 0.75; i++){
			//reading ith line
			lodo y_real = xt::row(data,i)[5];

			xt::xarray<lodo> row = xt::view(data, xt::keep(i), xt::range(0,5));
			tensor X (row);
			printf("X shape %d %d\n", X.val().shape(0), X.val().shape(1));
			cout << X << endl;

			tensor Y = xtad::dot(X,M) + C;

			tensor Loss = xtad::pow(xtad::pow( Y - y_real, 2 ), 0.5);

			cout << Loss <<endl;
			cout << M << endl;
			cout << C << endl;

			Loss.backward();

			cout << Loss <<endl;
			cout << M << endl;
			cout << C << endl;

			return 0;

			M = M + M.grad() * -1 * learning_rate;
			C = C + C.grad() * -1 * learning_rate;

			tensor::clear_buffer();
		}
		cout << epoch << " : " << M.val() << "   " << C.val() << endl ;
	}

	cout << "\nPredicted Y vs Actual Y" << endl;
	tensor rmse {0};
	int test_count {0};

	for(size_t i = N * 0.75; i < N; i++){
		test_count++;
			lodo y_real = xt::row(data,i)[5];

			xt::xarray<lodo> row = xt::view(data, xt::keep(i), xt::range(0,5));
			tensor X (row);

			tensor Y = xtad::dot(X,M) + C;

		cout << Y.val() << ' ' << y_real << endl;
		rmse += xtad::pow(Y - y_real, 2);
	}
	rmse /= test_count;
	rmse = xtad::pow(rmse, 0.5);
	cout << "\nRoot Mean Square Error : " << rmse.val() << endl;
}
