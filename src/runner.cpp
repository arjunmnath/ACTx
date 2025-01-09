#include "tensor.mm"
#include <Foundation/Foundation.h>
#include <iostream>
#include <vector>
using namespace std;

int main() {
  std::vector<float> data2 = {1.2, 2.3, 3.6, 4.0, 5.9, 6.1, 7.4, 8.2, 9.3};
  Tensor<float> *mat_a = new Tensor<float>(data2, std::vector<int>{3, 3});
  std::vector<int> shape = {9, 3};
  /*Tensor<float> a = Tensor<float>::ones(shape);*/
  /*Tensor<float> b = Tensor<float>::eye(3);*/
  /*Tensor<float> c = Tensor<float>::full(shape, 4);*/
  /*Tensor<float> d = Tensor<float>::zeros(shape);*/
  /*Tensor<float> e = Tensor<float>::clone(mat_a);*/
  Tensor<float> f = Tensor<float>::randn(shape);
  Tensor<float> j = Tensor<float>::randn(shape);
  /*Tensor<float> l = Tensor<float>::poission(f);*/
  Tensor<float> re = f.logical_gte(&f);
  re.print_matrix();
  std::cout << re.all() << std::endl;
  /*a.print_matrix();*/
  /*b.print_matrix();*/
  /*c.print_matrix();*/
  /*d.print_matrix();*/
  /*mat_a->print_matrix();*/
  /*e.print_matrix();*/
  /*f.print_matrix();*/
  /*l.print_matrix();*/
  return 0;
}
