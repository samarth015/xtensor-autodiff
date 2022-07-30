# Xtensor Autodiff

An automatic differentiation extension for [xtensor](https://github.com/xtensor-stack/xtensor). Implemented as a thin wrapper around xt::xarray template class. The interface is the same as xt::xarray except that the operations on the tensors are recorded and the results can differentiated with respect to the operands by simply calling the backwards() method.

A multivariate linear regression model has also been implemented by utilizing this extension.

### Simple example
```c++
#include "xtad/tensor_autograd.cpp"

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
```

> value : {{ 5.}}  grad : {{ 410.}}<br>
> value : {{ 10.}} grad : {{ 55.}}<br>
> value : {{ 11.}} grad : {{ 50.}}<br>
> value : {{ 17.}} grad : {{ 20.}}<br>
> value : {{ 1390.}} grad :  1.<br>

