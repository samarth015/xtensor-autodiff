#include "../tensor_autograd.cpp"
#include <fstream>
#include <sstream>

using namespace std;
using tensor = xtad::xarray<double>;

int main(int argc, char **argv){

	auto filename = "data/data1";

	std::ifstream data_file {filename};
	std::string line;

	int N {0};   //number of samples
	while(getline(data_file, line)) N++;

	double learning_rate { 0.05 };
	tensor M = {1,1,1,1,1}, C = {1};

	size_t i;
	for(size_t j{1}; j <= 5; j++){
		data_file.clear(); data_file.seekg(0);   //go to begining of file

		for(i = 0; i < N * 0.75; i++){
			//reading ith line
			std::getline(data_file, line);
			std::stringstream linebuf {line};

			double x1, x2, x3, x4, x5, y_real;
			linebuf >> x1 >> x2 >> x3 >> x4 >> x5 >> y_real;

			tensor X = {x1, x2, x3, x4, x5};
			X.reshape({5,1});

			tensor Y = xtad::dot(M,X) + C;

			tensor Loss = xtad::pow( Y - y_real, 2 );

			Loss.backward();

			M = M + M.grad() * -1 * learning_rate;
			C = C + C.grad() * -1 * learning_rate;

			tensor::clear_buffer();
		}
		cout << j << " : " << M.val() << "   " << C.val() << endl ;
	}

	cout << "\nPredicted Y vs Actual Y" << endl;
	tensor rmse {0};
	int test_count {0};

	for(; i < N; i++){
		//reading ith line
		std::getline(data_file, line);
		std::stringstream linebuf {line};

		double x1, x2, x3, x4, x5, y_real;
		linebuf >> x1 >> x2 >> x3 >> x4 >> x5 >> y_real;

		tensor X = {x1, x2, x3, x4, x5};
		X.reshape({5,1});

		tensor Y = xtad::dot(M,X) + C;

		cout << Y.val() << ' ' << y_real << endl;
		rmse += xtad::pow(Y - y_real, 2);
	}
	rmse /= test_count;
	rmse = xtad::pow(rmse, 0.5);
	cout << "\nRoot Mean Square Error : " << rmse.val() << endl;
}
