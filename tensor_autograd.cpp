#include <math.h>
#include <algorithm>
#include "tensor_autograd.hpp"

namespace xtad {

	template<typename ftype>
		xarray<ftype>::node::node(_xarray &&val, parents par) noexcept : 
		grad{0},
		val{std::move(val)}, 
		par{par}, 
		eval_grad{ [](node*){} }
	{}

	template<typename ftype>
		xarray<ftype>::xarray(node* p): sh_ptr{ p } {
			global_buffer.push_back(sh_ptr);
		} 


	template<typename ftype>
		xarray<ftype>::xarray(): 
			xarray(new node{ _xarray{}, 
					parents{nullptr, nullptr}}) 
	{}

	template<typename ftype>
		xarray<ftype>::xarray(_xarray val): 
			xarray(new node{ std::move(val), 
					parents{nullptr, nullptr}}) 
	{}

	//static members
	template<typename ftype>
		typename xarray<ftype>::global_storage xarray<ftype>::global_buffer(0);  // empty initially

	//copy constructor
	template<typename ftype>
		xarray<ftype>::xarray(const xarray &other): 
			xarray( new node{  _xarray{other.sh_ptr->val},   // Creating a copy of _xarray to be consumed by node.
					parents{ other.sh_ptr.get(), nullptr } } ) 
	{
		sh_ptr->eval_grad = [](node *child) {
			child->par.first->grad += child->grad;
		};
	}

	//move constructor
	template<typename ftype>
		xarray<ftype>::xarray(xarray &&other) noexcept: sh_ptr{other.sh_ptr} {
			other.sh_ptr.reset();
		}

	//copy assignment
	template<typename ftype>
		xarray<ftype>& xarray<ftype>::operator = ( const xarray &other){
			if(this == &other) return *this;
			*this = xarray{other};   // calls copy constructor followed by move assignment overload
			return *this;
		}

	//move constructor
	template<typename ftype>
		xarray<ftype> & xarray<ftype>::operator = (xarray &&other) noexcept {
			this->sh_ptr = other.sh_ptr;       // not checking for equality of lhs and rhs because its fast enough already
			other.sh_ptr = nullptr;       
			return *this;
		}

	template<typename ftype>
		xarray<ftype> xarray<ftype>::operator + (const xarray &oth) const {

			_xarray result { this->sh_ptr->val + oth.sh_ptr->val };
			node* result_node { new node{  std::move(result), parents{this->sh_ptr.get(), oth.sh_ptr.get()} } };

			result_node->eval_grad = [](node *child) {
				child->par.first->grad += child->grad;
				child->par.second->grad += child->grad;
			};

			return xarray{result_node};
		}

	template<typename ftype>
		xarray<ftype> xarray<ftype>::operator * (const xarray &oth) const {

			_xarray result { this->sh_ptr->val * oth.sh_ptr->val };
			node* result_node = new node{ std::move(result), {this->sh_ptr.get(), oth.sh_ptr.get()} };

			result_node->eval_grad = [] (node *child) {
				child->par.first->grad += child->par.second->val * child->grad;
				child->par.second->grad += child->par.first->val * child->grad;
			};

			return xarray{result_node};
		}

	template<typename ftype>
		xarray<ftype> xarray<ftype>::operator - () const {
			return -1 * (*this);
		}

	template<typename ftype>
		xarray<ftype> xarray<ftype>::operator - (const xarray &oth) const {
			return (*this) + (-oth);
		}

	template<typename ftype>
		xarray<ftype> xarray<ftype>::operator / (const xarray &oth) const{
			xarray mul_inverse { pow(oth, -1) };
			return (*this) * mul_inverse;
		}

	template<typename ftype>
		bool xarray<ftype>::operator == (const xarray &oth) const {
			return this->sh_ptr->val == oth.sh_ptr->val;
		}

	template<typename ftype>
		bool xarray<ftype>::operator != (const xarray &oth) const {
			return not(*this == oth);
		}

	template<typename ftype>
		xarray<ftype> & xarray<ftype>::operator += (const xarray &oth){
			*this = *this + oth;
			return *this;
		}

	template<typename ftype>
		xarray<ftype> & xarray<ftype>::operator -= (const xarray &oth){
			*this = *this + (-oth);
			return *this;
		}

	template<typename ftype>
		xarray<ftype> & xarray<ftype>::operator *= (const xarray &oth){
			*this = *this * oth;
			return *this;
		}

	template<typename ftype>
		xarray<ftype> & xarray<ftype>::operator /= (const xarray &oth){
			*this = *this / oth;
			return *this;
		}

	template<typename ftype, typename T>
		xarray<ftype> operator + (T num, xarray<ftype> rn){
			return xarray<ftype>{num} + rn;
		}

	template<typename ftype, typename T>
		xarray<ftype> operator + (xarray<ftype> rn, T num){
			return xarray<ftype>{num} + rn;
		}

	template<typename ftype, typename T>
		xarray<ftype> operator - (T num, xarray<ftype> rn){
			return xarray<ftype>{num} - rn;
		}

	template<typename ftype, typename T>
		xarray<ftype> operator - (xarray<ftype> rn, T num){
			return rn - xarray<ftype>{num};
		}

	template<typename ftype, typename T>
		xarray<ftype> operator * (T num, xarray<ftype> rn){
			return xarray<ftype>{num} * rn;
		}

	template<typename ftype, typename T>
		xarray<ftype> operator * (xarray<ftype> rn, T num){
			return xarray<ftype>{num} * rn;
		}

	template<typename ftype, typename T>
		xarray<ftype> operator / (T num, xarray<ftype> rn){
			return xarray<ftype>{num} / rn;
		}

	template<typename ftype, typename T>
		xarray<ftype> operator / (xarray<ftype> rn, T num){
			return rn / xarray<ftype>{num};
		}

	template<typename ftype, typename T>
		xarray<ftype> & operator += (xarray<ftype> &rn, T num){
			rn += xarray<ftype>{num};
			return rn;
		}

	template<typename ftype, typename T>
		xarray<ftype> & operator -= (xarray<ftype> &rn, T num){
			rn -= xarray<ftype>{num};
			return rn;
		}

	template<typename ftype, typename T>
		xarray<ftype> & operator *= (xarray<ftype> &rn, T num){
			rn *= xarray<ftype>{num};
			return rn;
		}

	template<typename ftype, typename T>
		xarray<ftype> & operator /= (xarray<ftype> &rn, T num){
			rn /= xarray<ftype>{num};
			return rn;
		}

	template<typename ftype>
		xarray<ftype> pow(const xarray<ftype> &a, const xarray<ftype> &b){    
			// Also had to mark const to _xarray methods 
			using _xarray = xt::xarray<ftype>;
			using node = typename xarray<ftype>::node;

			_xarray result { xt::pow( a.sh_ptr->val, b.sh_ptr->val ) };
			node* result_node = new node{ std::move(result), {a.sh_ptr.get(), nullptr} };

			result_node->eval_grad = [second_val = b.sh_ptr->val] (node *child){
				child->par.first->grad += second_val * xt::pow(child->par.first->val, second_val - _xarray{1}) * child->grad;
			};

			return xarray<ftype>{result_node};
		}

	template<typename ftype, typename T>
		xarray<ftype> pow(const xarray<ftype> &a, T b){    
			return pow( a, xarray<ftype>{ static_cast<ftype>(b) } );
		}

	template<typename ftype, typename T>
		xarray<ftype> pow(T a, const xarray<ftype> &b){   
			return pow( xarray<ftype>{ a }, static_cast<ftype>(b) );
		}

	template<typename ftype>
		xarray<ftype> dot(const xarray<ftype> &a, const xarray<ftype>& b){
			using _xarray = xt::xarray<ftype>;
			using node = typename xarray<ftype>::node;

			_xarray result { xt::linalg::dot( a.sh_ptr->val, b.sh_ptr->val ) };
			node* result_node = new node{ std::move(result), {a.sh_ptr.get(), b.sh_ptr.get()} };

			result_node->eval_grad = [] (node *child){
				child->par.first->grad += xt::transpose(child->par.second->val) * child->grad;
				child->par.second->grad += xt::transpose(child->par.first->val) * child->grad;
			};

			return xarray<ftype>{result_node};
		}

	/*
	   template<typename ftype>
	   xarray<ftype> relu(const xarray<ftype> &num){
	   using _xarray = xt::xarray<ftype>;
	   using node = typename xarray<ftype>::node;

	   _xarray result { xt::relu( num.sh_ptr->val ) };
	   node* result_node = new node{ std::move(result), {num.sh_ptr.get(), nullptr} };

	   result_node->eval_grad = [](node *child){
	   child->par.first->grad += _xarray{ (child->val == 0 ? 0.0 : 1.0) } * child->grad;  // improve this line
	   };

	   return xarray<ftype>{result_node};

	   }
	   */

	/*
	   template<typename ftype, typename T>
	   xarray<ftype> relu(T num){
	   return relu( xarray<ftype>{num} );
	   }
	   */

	template<typename ftype>
		void xarray<ftype>::build_topo(node *no, std::vector<node*> &toporder, std::vector<node*> &visited){   
			if(std::find(visited.begin(), visited.end(), no) == visited.end()){
				visited.push_back(no);              
				if(no->par.first != nullptr) build_topo(no->par.first, toporder, visited);
				if(no->par.second != nullptr) build_topo(no->par.second, toporder, visited);
				toporder.push_back(no);              
			}
		}

	template<typename ftype>
		void xarray<ftype>::backward(){
			std::vector<node*> toporder{}, visited{}; 
			build_topo(sh_ptr.get(), toporder, visited);

			for(node* no : toporder) no->grad = 0;

			sh_ptr->grad = 1;

			std::vector<node*> reversed_toporder {toporder.rbegin(), toporder.rend()};

			for(node* no : reversed_toporder) no->eval_grad(no);
		}

	template<typename ftype>
		std::ostream & operator << ( std::ostream &output, const xarray<ftype> &num ){
			output << "value : " << num.sh_ptr->val << std::endl
				   << "grad : " << num.sh_ptr->grad << std::endl;
			return output;
		}

	template<typename ftype>
		void xarray<ftype>::clear_buffer(){
			// need to reset parents of named variables
			//for(auto a : global_buffer) std::cout << a.use_count() << ' ' << a->val << std::endl ;	
			// use count for intermediates nodes is 2 since we have 1 in buffer and another is a (lambda param). Count is 
			// 3 for named xarray because of the additional sh_ptr.
			auto it = std::remove_if(global_buffer.begin(), global_buffer.end(), [](auto a){ return a.use_count() == 2; })	;
			global_buffer.erase(it, global_buffer.end());
			for(auto shptr : global_buffer){
				shptr->par = parents{nullptr, nullptr};
				shptr->eval_grad = [](node*){}; 
			}
		}

	/*
	   template<typename Scalar,        //xarray instantiated type
	   typename ...Args_Type>

	   std::tuple<Scalar, Scalar, Args_Type...> 
	   get_gradScalar( Scalar(*func)(Scalar, Args_Type...), Scalar x, Args_Type ...args)
	   {
	   Scalar result = func(x ,args... );
	   result.backward();
	   auto args_list = {x ,args...};
	   std::tuple<Scalar, Scalar, Args_Type...> gradients;
	   gradients[0] = std::move(result);
	   gradients[1] = x.grad;
	   for(std::size_t i{0}; i < sizeof(args_list); i++){
	   gradients[i+2] = args_list[i].grad;
	   }
	   return gradients;
	   }
	   */
}
