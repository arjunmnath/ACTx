
#include "dispatcher.h"
#include <gtest/gtest.h>

Tensor *make_tensor(std::vector<float> data, std::vector<int> shape = {2}) {
  Tensor *t = new Tensor(data, shape, DType::float32, false);
  return t;
}

TEST(DispatcherTest, AddOperationMPS) {
  Dispatcher dispatcher;
  dispatcher.init_register();

  Tensor *a = make_tensor({1.0f, 2.0f});
  Tensor *b = make_tensor({3.0f, 4.0f});
  Tensor *result = make_tensor({0.0f, 0.0f});

  dispatcher.call(OPType::ADD, DeviceType::MPS, a, b, result);

  EXPECT_EQ(result->getElement(0), 4.0f);
  EXPECT_EQ(result->getElement(1), 6.0f);
}

TEST(DispatcherTest, SubOperationMPS) {
  Dispatcher dispatcher;
  dispatcher.init_register();

  Tensor *a = make_tensor({5.0f, 7.0f});
  Tensor *b = make_tensor({3.0f, 2.0f});
  Tensor *result = make_tensor({0.0f, 0.0f});

  dispatcher.call(OPType::SUB, DeviceType::MPS, a, b, result);

  EXPECT_EQ(result->getElement(0), 2.0f);
  EXPECT_EQ(result->getElement(1), 5.0f);
}

TEST(DispatcherTest, MulOperationMPS) {
  Dispatcher dispatcher;
  dispatcher.init_register();

  Tensor *a = make_tensor({2.0f, 3.0f});
  Tensor *b = make_tensor({4.0f, 5.0f});
  Tensor *result = make_tensor({0.0f, 0.0f});

  dispatcher.call(OPType::MUL, DeviceType::MPS, a, b, result);

  EXPECT_EQ(result->getElement(0), 8.0f);
  EXPECT_EQ(result->getElement(1), 15.0f);
}

TEST(DispatcherTest, DivOperationMPS) {
  Dispatcher dispatcher;
  dispatcher.init_register();

  Tensor *a = make_tensor({10.0f, 20.0f});
  Tensor *b = make_tensor({2.0f, 5.0f});
  Tensor *result = make_tensor({0.0f, 0.0f});

  dispatcher.call(OPType::DIV, DeviceType::MPS, a, b, result);

  EXPECT_EQ(result->getElement(0), 5.0f);
  EXPECT_EQ(result->getElement(1), 4.0f);
}

/*TEST(DispatcherTest, MatMulOperationMPS) {*/
/*  Dispatcher dispatcher;*/
/*  dispatcher.init_register();*/
/**/
/*  Tensor a = make_tensor({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});*/
/*  Tensor b = make_tensor({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2});*/
/*  Tensor result = make_tensor({0.0f, 0.0f, 0.0f, 0.0f}, {2, 2});*/
/**/
/*  dispatcher.call(OPType::MATMUL, DeviceType::MPS, a, b, result);*/
/**/
/*  EXPECT_FLOAT_EQ(result.getElement(0), 19.0f);*/
/*  EXPECT_FLOAT_EQ(result.getElement(1), 22.0f);*/
/*  EXPECT_FLOAT_EQ(result.getElement(2), 43.0f);*/
/*  EXPECT_FLOAT_EQ(result.getElement(3), 50.0f);*/
/*}*/
