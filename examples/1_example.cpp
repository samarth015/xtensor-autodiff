#include "../tensor_autograd.cpp"

using namespace std;
using tensor = xtad::xarray<double>;

int main(){

	tensor x = {10};
	tensor y = 7 * x + xtad::pow(x, 3);   // y is a function of x
	
	y.backward();     // Calculating the derivative

	cout << x << y;
}
