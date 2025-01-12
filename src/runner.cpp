#include "tensor.mm"
#include <Foundation/Foundation.h>
#include <iostream>
#include <vector>
using namespace std;

int main() {
  /*Tensor<float> a = Tensor<float>::ones(shape);*/
  /*Tensor<float> b = Tensor<float>::eye(3);*/
  /*Tensor<float> c = Tensor<float>::full(shape, 4);*/
  /*Tensor<float> d = Tensor<float>::zeros(shape);*/
  /*Tensor<float> e = Tensor<float>::clone(mat_a);*/
  /*Tensor<float> l = Tensor<float>::poission(f);*/
  /*a.print_matrix();*/
  /*b.print_matrix();*/
  /*c.print_matrix();*/
  /*d.print_matrix();*/
  /*mat_a->print_matrix();*/
  /*e.print_matrix();*/
  /*f.print_matrix();*/
  /*l.print_matrix();*/

  std::vector<int> shape = {1000, 1000};
  Tensor<float> mat_a = Tensor<float>::full(shape, 2);
  Tensor<float> mat_b = Tensor<float>::full(shape, 2);
  Tensor<float> mat_c = Tensor<float>::full(shape, 1);
  Tensor<float> mat_d = Tensor<float>::full(shape, 0.1f);
  mat_c.exp(true);
  for (int i = 0; i < 10000; i++) {
    mat_a.mul(&mat_b, true);
    mat_a.div(&mat_c, true);
    mat_a.add(&mat_d, true);
  }
  mat_a.print_matrix();
  return 0;
}
