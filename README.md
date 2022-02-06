# Xtensor Autodiff

An automatic differentiation extension for [xtensor](https://github.com/xtensor-stack/xtensor). Implemented as a thin wrapper around xt::xarray template class. The interface is the same as xt::xarray except that the operations on the tensors are recorded and the results can differentiated with respect to the operands by simply calling the backwards() method.
