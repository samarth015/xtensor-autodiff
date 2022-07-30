#include "../xtad/tensor_autograd.cpp"

using namespace std;
using tensor = xtad::xarray<double>;

int main(){

	tensor p = {5};
	tensor q = {10};
	tensor r = {11};
	tensor s = {17};

	tensor z = xtad::pow(p, 3) * 4 + q * r * p + 20 * s;
	
	z.backward();     // Calculating the derivative

	cout << p << q << r << s << z;
}
