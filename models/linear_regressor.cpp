#include "../xtad/tensor_autograd.cpp"
#include <xtensor/xcsv.hpp>
#include <vector>

using namespace std;
using lodo = long double;
using tensor = xtad::xarray<lodo>;

template<typename T>
class linear_regressor{
	private:
		xt::xarray<T> coeff;
		xt::xarray<T> bias;

	public:
		linear_regressor() = default;

		void fit(const xt::xarray<T> &X, const xt::xarray<T> &y, lodo learning_rate, size_t epochs){
			size_t M = X.shape(0), N = X.shape(1);   //number of training samples and no of features

			tensor coeff (xt::random::rand<T>({(int)N,1})), 
				   C  (xt::random::rand<T>({1,1}));

			for(size_t epoch{1}; epoch <= epochs; epoch++){
				for(size_t i = 0; i < M; i++){

					lodo y_real = y[i];
					xt::xarray<lodo> row = xt::row(X, i);
					// coeff is represented as a 2D matrix of 1 col so representing
					// X as a 2D matrix of 1 row
					row.reshape({1,N}); 
					tensor X (row);
					
					tensor Y = xtad::dot(X,coeff) + C;

					tensor Loss = xtad::pow( Y - y_real, 2 );

					Loss.backward();

					coeff = coeff + coeff.grad() * -1 * learning_rate;
					C = C + C.grad() * -1 * learning_rate;

					tensor::clear_buffer();

				}
				cout << xt::flatten(coeff.val()) << ' ' << C.val() << endl;
			}

			this->coeff = coeff.val();
			this->bias = C.val();
		}

		const xt::xarray<T>& coefficients() const {
			return coeff;
		}

		xt::xarray<T> biases() {
			return bias;
		}

		xt::xarray<T> predict(const xt::xarray<T>& X){
			if(X.dimension() == 1){
				xt::xarray<T> X_copy = X;
				int N = X.size();
				X_copy.reshape({1,N});
				return xt::transpose(xt::linalg::dot(X_copy, coeff) + bias);
			}
			else{
				return xt::transpose(xt::linalg::dot(X, coeff) + bias);
			}
		}
};
