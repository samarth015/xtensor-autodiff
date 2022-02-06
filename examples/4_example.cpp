
#include "../tensor_autograd.cpp"

using namespace std;
using tensor = xtad::xarray<double>;

int main(){

	tensor A = {{1,2,3},
			    {3,4,5},
			    {7,8,9}};

	tensor B = {{1,2,3},
			    {3,4,5},
			    {7,8,9}};

	tensor C = xtad::dot(A,B);  // dz/dp = transpose(q) and vice versa
	
	C.backward();     

	cout << A << B << C;
}
