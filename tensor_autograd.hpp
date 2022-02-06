#ifndef TENSOR_AUTODIFF__H_
#define TENSOR_AUTODIFF__H_

#include <xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include <utility>
#include <vector>
#include <memory>
#include <functional>
#include <array>
#include <tuple>
#include <algorithm>

namespace xtad {

	template<typename ftype = double>
		class xarray{

			private:
				class node;
				using _xarray = xt::xarray<ftype>;  // inner xarray
				using parents = std::pair<node*, node*>; // pair of parent nodes.

				class node
				{
					public:
						_xarray grad;
						_xarray val;
						parents par;
						std::function<void(node*)> eval_grad;  //backwards()

						node(_xarray &&val, parents par) noexcept;

						node(node &&other) noexcept = delete;
						node(const node &other) = delete;
						node() = delete;

						~node() = default;
				};

				//node is deallocated when 1) pointer is removed from both global storage 2)xarray objects is deleted
			public: std::shared_ptr<node> sh_ptr; 

				// containter to save intermediate values and deallocate them automatically.
				using global_storage = std::vector<std::shared_ptr<node>>; 
				static global_storage global_buffer;

				xarray(node* ptr);

			public:

				xarray();
				xarray(_xarray);
				
				xarray(xt::nested_initializer_list_t<ftype,1> li): xarray( new node{ _xarray{li}, parents{nullptr, nullptr}}) {} 
				xarray(xt::nested_initializer_list_t<ftype,2> li): xarray( new node{ _xarray{li}, parents{nullptr, nullptr}}) {} 
				xarray(xt::nested_initializer_list_t<ftype,3> li): xarray( new node{ _xarray{li}, parents{nullptr, nullptr}}) {} 
				xarray(xt::nested_initializer_list_t<ftype,4> li): xarray( new node{ _xarray{li}, parents{nullptr, nullptr}}) {} 
				xarray(xt::nested_initializer_list_t<ftype,5> li): xarray( new node{ _xarray{li}, parents{nullptr, nullptr}}) {} 

				xarray(const xarray &other);
				xarray(xarray &&other) noexcept;
				~xarray() = default;

				xarray & operator = ( const xarray &other);
				xarray & operator = (xarray &&other) noexcept ;

				xarray operator + (const xarray &oth) const ;
				xarray operator * (const xarray &oth) const ;
				xarray operator - () const ;
				xarray operator - (const xarray &oth) const ;
				xarray operator / (const xarray &oth) const ;

				bool operator == (const xarray &oth) const ;
				bool operator != (const xarray &oth) const ;

				xarray & operator += (const xarray &oth);
				xarray & operator -= (const xarray &oth);
				xarray & operator *= (const xarray &oth);
				xarray & operator /= (const xarray &oth);

				template<typename T>
					friend std::ostream &operator<<( std::ostream &output, const xarray<T> &num );  			
				template<typename T>
					friend xarray<T> pow(const xarray<T>&, const xarray<T>&);
				template<typename T>
					friend xarray<T> dot(const xarray<T> &, const xarray<T>&);

				// Delare the second overload of relu here
				/*
				   template<typename T>
				   friend xarray<T> relu(const xarray<T>&);
				   */

				static void clear_buffer();

				void build_topo(node *no, std::vector<node*> &toporder, std::vector<node*> &visited);   

				void backward();

				const _xarray& grad() const {
					return sh_ptr->grad;
				}

				const _xarray& val() const {
					return sh_ptr->val;
				}

				void reshape(std::initializer_list<int> li){
					sh_ptr->val.reshape(li);
				}
		};

	template<typename ftype,typename Num>
		xarray<ftype> operator + (Num, xarray<ftype>);          

	template<typename ftype,typename Num>
		xarray<ftype> operator + (xarray<ftype>, Num);           

	template<typename ftype,typename Num>
		xarray<ftype> operator - (Num, xarray<ftype>);

	template<typename ftype,typename Num>
		xarray<ftype> operator - (xarray<ftype>, Num);

	template<typename ftype,typename Num>
		xarray<ftype> operator * (Num, xarray<ftype>);

	template<typename ftype,typename Num>
		xarray<ftype> operator * (Num, xarray<ftype>);

	template<typename ftype,typename Num>
		xarray<ftype> operator / (xarray<ftype>, Num);

	template<typename ftype,typename Num>
		xarray<ftype> operator / (xarray<ftype>, Num);

	template<typename ftype,typename Num>
		xarray<ftype> & operator += (xarray<ftype>&, Num);

	template<typename ftype,typename Num>
		xarray<ftype> & operator -= (xarray<ftype>&, Num);

	template<typename ftype,typename Num>
		xarray<ftype> & operator *= (xarray<ftype>&, Num);

	template<typename ftype,typename Num>
		xarray<ftype> & operator /= (xarray<ftype>&, Num);

	template<typename ftype,typename Num>
		xarray<ftype> relu(Num);

	template<typename ftype,typename Num>
		xarray<ftype> pow(const xarray<ftype> &a, Num b);


	/*
	   template<typename Return_Type, typename ...Arg_Types>
	   class get_gradient
	   {
	   private:
	   Return_Type(*func)( Arg_Types... );

	   public:
	   get_gradient() = delete;
	   get_gradient(const get_gradient &other) = default;
	   get_gradient(get_gradient &&other) noexcept = default;
	   ~get_gradient() = default;

	   get_gradient(Return_Type(*func)( Arg_Types... )): func{func} {} 

	   std::vector<Return_Type> 
	   operator    ()    (Arg_Types ...args) {

	   Return_Type result { func(args...) };
	   result.backward();

	   auto args_list = {result ,args...};
	   std::vector<Return_Type> gradients;

	   gradients[0] = std::move(result);
	   int i {1};
	   for(auto arg : args_list){
	   gradients[1] = arg.grad;
	   i++;
	   }

	   return gradients;
	   }
	   };
	   */

	/*
	   template<typename Scalar,        //xarray instantiated type
	   typename ...Args_Type>
	   std::tuple<Scalar, Scalar, Args_Type...> 
	   get_gradScalar( Scalar(*func)(Scalar, Args_Type...), Scalar x, Args_Type ...args);
	   */

} 

//#include "tensor_autograd.cpp"

#endif
