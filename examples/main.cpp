#include "tensor_autograd.cpp"

using namespace std;
using tensor = xtad::xarray<double>;
using _t = xt::xarray<double>;

int main(){
	tensor x ({{1,2,3},{1,2,3},{1,2,3}});
	tensor y ({{2,3,5},{3,5,7},{10,10,10}});
	//tensor z = x * y;
	//tensor z = (x*y + xtad::pow(x, 2) * y )*y;
	
	
	tensor z = xtad::dot(x,y);
	z.backward();

	cout << z << endl << x << endl << y << endl ;
	//tensor z = (x*y + xtad::pow(z, 2) * y )*y;
}
