#include "../xtad/tensor_autograd.cpp"

using namespace std;
using tensor = xtad::xarray<double>;

int main(){

	tensor p = {1,2,3,5};
	tensor q = {4,5,6,9};

	tensor z = 3 * p + q;  // dz/dp = transpose(q) and vice versa
	
	z.backward();     

	cout << p << q << z;
}
