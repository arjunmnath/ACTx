#include "dispatcher.h"
#include "device_type.h"
#include "main.h"
#include "op_types.h"
#include "opnode.h"
#include "tensor.h"
#include <cmath>
#include <iostream>
#include <optional>
#include <stdexcept>

#define REGISTER_OP(OP, DEVICE, FUNC_PRE, FUNC_POST, BACKWARD)                 \
  this->_register->register_op(                                                \
      OPType::OP, DeviceType::DEVICE,                                          \
      [this](std::vector<Tensor *> inputs) -> void {                           \
        Tensor *a, *b, *result;                                                \
        a = b = result = nullptr;                                              \
        FUNC_PRE;                                                              \
        if (a && b) {                                                          \
          a->requires_grad = b->requires_grad =                                \
              a->requires_grad || b->requires_grad;                            \
        }                                                                      \
        if (result) {                                                          \
          result->node = new OpNode;                                           \
          result->node->op =                                                   \
              this->_register->get(OPType::OP, DeviceType::DEVICE);            \
          result->node->type = OPType::OP;                                     \
        }                                                                      \
        FUNC_POST;                                                             \
      },                                                                       \
      [](OpNode *node) -> void {                                               \
        Tensor *a, *b, *out;                                                   \
        a = b = out = nullptr;                                                 \
        BACKWARD;                                                              \
      })

void Dispatcher::call(OPType op, DeviceType device,
                      std::vector<Tensor *> inputs) {
  Operation *operation = this->_register->get(op, device);
  if (operation == nullptr) {
    throw std::logic_error("operation not found");
  }
  operation->func(inputs);
}

Operation *Dispatcher::get(OPType op, DeviceType device) {
  return this->_register->get(op, device);
}

void Dispatcher::init_register() {
  REGISTER_OP(NEGATE, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                result = inputs[1];
              }),
              ({
                result->node->inputs = {a};
                result->node->outputs = {result};
                mps->negate(a, result);
              }),
              {
                a = node->inputs[0];
                out = node->outputs[0];
                assert(node->inputs.size() == 1 && node->outputs.size() == 1);
                if (a->requires_grad)
                  a->grad = a->grad != nullptr
                                ? a->grad->add(out->grad->negate(false), true)
                                : Tensor::clone(out->grad)->negate(false);
              });

  REGISTER_OP(ADD, MPS, ({
                assert(inputs.size() == 3);
                a = inputs[0];
                b = inputs[1];
                result = inputs[2];
              }),
              ({
                result->node->inputs = {a, b};
                result->node->outputs = {result};
                mps->add(a, b, result);
              }),
              ({
                assert(node->inputs.size() == 2 && node->outputs.size() == 1);
                a = node->inputs[0];
                b = node->inputs[1];
                out = node->outputs[0];
                if (a->requires_grad)
                  a->grad = a->grad != nullptr ? a->grad->add(out->grad, true)
                                               : Tensor::clone(out->grad);
                if (b->requires_grad)
                  b->grad = b->grad != nullptr ? b->grad->add(out->grad)
                                               : Tensor::clone(out->grad);
              }));
  REGISTER_OP(SUB, MPS, ({
                assert(inputs.size() == 3);
                a = inputs[0];
                b = inputs[1];
                result = inputs[2];
              }),
              ({
                result->node->inputs = {a, b};
                result->node->outputs = {result};
                mps->sub(a, b, result);
              }),
              {
                assert(node->inputs.size() == 2 && node->outputs.size() == 1);
                a = node->inputs[0];
                b = node->inputs[1];
                out = node->outputs[0];
                if (a->requires_grad)
                  a->grad = a->grad != nullptr ? a->grad->add(out->grad, true)
                                               : Tensor::clone(out->grad);
                if (b->requires_grad)
                  b->grad = b->grad != nullptr
                                ? b->grad->add(out->grad->negate(false), true)
                                : Tensor::clone(out->grad)->negate(false);
              });
  REGISTER_OP(MUL, MPS, ({
                assert(inputs.size() == 3);
                a = inputs[0];
                b = inputs[1];
                result = inputs[2];
              }),
              ({
                result->node->inputs = {a, b};
                result->node->outputs = {result};
                mps->mul(a, b, result);
              }),
              ({
                assert(node->inputs.size() == 2 && node->outputs.size() == 1);
                a = node->inputs[0];
                b = node->inputs[1];
                out = node->outputs[0];
                if (a->requires_grad)
                  a->grad = a->grad ? a->grad->add(b->mul(out->grad), true)
                                    : b->mul(Tensor::clone(out->grad));
                if (b->requires_grad)
                  b->grad = b->grad ? b->grad->add(a->mul(out->grad), true)
                                    : a->mul(Tensor::clone(out->grad));
              }));

  REGISTER_OP(DIV, MPS, ({
                assert(inputs.size() == 3);
                a = inputs[0];
                b = inputs[1];
                result = inputs[2];
              }),
              ({
                result->node->inputs = {a, b};
                result->node->outputs = {result};
                mps->div(a, b, result);
              }),
              ({
                assert(node->inputs.size() == 2 && node->outputs.size() == 1);
                a = node->inputs[0];
                b = node->inputs[1];
                out = node->outputs[0];
                if (a->requires_grad)
                  a->grad = a->grad
                                ? a->grad->add(out->grad->div(b, false), true)
                                : Tensor::clone(out->grad)->div(b, false);
                if (b->requires_grad)
                  b->grad =
                      b->grad ? b->grad->add(a->div(b->pow(2.0f, false), false)
                                                 ->mul(out->grad, false)
                                                 ->negate(false),
                                             true)
                              : a->div(b->pow(2.0f, false), false)
                                    ->mul(Tensor::clone(out->grad), false)
                                    ->negate(false);
              }));
  REGISTER_OP(
      POW, MPS, ({
        assert(inputs.size() == 3);
        a = inputs[0];
        b = inputs[1];
        result = inputs[2];
        assert(b->size == 1);
      }),
      ({
        result->node->inputs = {a, b};
        result->node->outputs = {result};
        mps->pow(a, b, result);
      }),
      ({
        assert(node->inputs.size() == 2 && node->outputs.size() == 1);
        a = node->inputs[0];
        b = node->inputs[1];
        out = node->outputs[0];
        if (a->requires_grad)
          a->grad =
              a->grad
                  ? a->grad->add(
                        b->mul(a->pow(b->_get_element(0) - 1))->mul(out->grad),
                        true)
                  : b->mul(a->pow(b->_get_element(0) - 1))
                        ->mul(Tensor::clone(out->grad));
      }));

  // comparison;
  REGISTER_OP(LOGICAL_E, MPS, ({
                assert(inputs.size() == 3);
                a = inputs[0];
                b = inputs[1];
                result = inputs[2];
              }),
              ({
                result->node->inputs = {a, b};
                result->node->outputs = {result};
                mps->logical_e(a, b, result);
              }),
              {
                out = node->outputs[0];
                if (out->requires_grad)
                  throw std::logic_error(
                      "Cannot attach comparison operations to compute graphs");
              });

  REGISTER_OP(LOGICAL_NE, MPS, ({
                assert(inputs.size() == 3);
                a = inputs[0];
                b = inputs[1];
                result = inputs[2];
              }),
              ({
                result->node->inputs = {a, b};
                result->node->outputs = {result};
                mps->logical_ne(a, b, result);
              }),
              {
                out = node->outputs[0];
                if (out->requires_grad)
                  throw std::logic_error(
                      "Cannot attach comparison operations to compute graphs");
              });

  REGISTER_OP(LOGICAL_GT, MPS, ({
                assert(inputs.size() == 3);
                a = inputs[0];
                b = inputs[1];
                result = inputs[2];
              }),
              ({
                result->node->inputs = {a, b};
                result->node->outputs = {result};
                mps->logical_gt(a, b, result);
              }),
              {
                out = node->outputs[0];
                if (out->requires_grad)
                  throw std::logic_error(
                      "Cannot attach comparison operations to compute graphs");
              });

  REGISTER_OP(LOGICAL_GTE, MPS, ({
                assert(inputs.size() == 3);
                a = inputs[0];
                b = inputs[1];
                result = inputs[2];
              }),
              ({
                result->node->inputs = {a, b};
                result->node->outputs = {result};
                mps->logical_gte(a, b, result);
              }),
              {
                out = node->outputs[0];
                if (out->requires_grad)
                  throw std::logic_error(
                      "Cannot attach comparison operations to compute graphs");
              });

  REGISTER_OP(LOGICAL_LTE, MPS, ({
                assert(inputs.size() == 3);
                a = inputs[0];
                b = inputs[1];
                result = inputs[2];
              }),
              ({
                result->node->inputs = {a, b};
                result->node->outputs = {result};
                mps->logical_lte(a, b, result);
              }),
              {
                out = node->outputs[0];
                if (out->requires_grad)
                  throw std::logic_error(
                      "Cannot attach comparison operations to compute graphs");
              });

  REGISTER_OP(LOGICAL_LT, MPS, ({
                assert(inputs.size() == 3);
                a = inputs[0];
                b = inputs[1];
                result = inputs[2];
              }),
              ({
                result->node->inputs = {a, b};
                result->node->outputs = {result};
                mps->logical_lt(a, b, result);
              }),
              {
                out = node->outputs[0];
                if (out->requires_grad)
                  throw std::logic_error(
                      "Cannot attach comparison operations to compute graphs");
              });

  // math functions;
  REGISTER_OP(
      SQRT, MPS, ({
        assert(inputs.size() == 2);
        a = inputs[0];
        result = inputs[1];
      }),
      ({
        result->node->inputs = {a};
        result->node->outputs = {result};
        mps->sqrt(a, result);
      }),
      {
        a = node->inputs[0];
        out = node->outputs[0];
        assert(node->inputs.size() == 1 && node->outputs.size() == 1);
        if (a->requires_grad)
          a->grad =
              a->pow(-0.5f)->div(Tensor::full_like(a, 2.0f))->mul(out->grad);
      });

  REGISTER_OP(EXP, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                result = inputs[1];
              }),
              ({
                result->node->inputs = {a};
                result->node->outputs = {result};
                mps->exp(a, result);
              }),
              {
                a = node->inputs[0];
                out = node->outputs[0];
                assert(node->inputs.size() == 1 && node->outputs.size() == 1);
                if (a->requires_grad)
                  a->grad = a->exp()->mul(out->grad);
              });

  REGISTER_OP(LOG, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                result = inputs[1];
              }),
              ({
                result->node->inputs = {a};
                result->node->outputs = {result};
                mps->log(a, result);
              }),
              {
                a = node->inputs[0];
                out = node->outputs[0];
                assert(node->inputs.size() == 1 && node->outputs.size() == 1);
                if (a->requires_grad)
                  a->grad = Tensor::ones_like(a)->div(a)->mul(out->grad);
              });
  REGISTER_OP(LOG10, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                result = inputs[1];
              }),
              ({
                result->node->inputs = {a};
                result->node->outputs = {result};
                mps->log10(a, result);
              }),
              {
                a = node->inputs[0];
                out = node->outputs[0];
                assert(node->inputs.size() == 1 && node->outputs.size() == 1);
                if (a->requires_grad)
                  a->grad = Tensor::full_like(a, 1.0f / (float)log(10))
                                ->div(a)
                                ->mul(out->grad);
              });

  REGISTER_OP(LOG2, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                result = inputs[1];
              }),
              ({
                result->node->inputs = {a};
                result->node->outputs = {result};
                mps->log2(a, result);
              }),
              {
                a = node->inputs[0];
                out = node->outputs[0];
                assert(node->inputs.size() == 1 && node->outputs.size() == 1);
                if (a->requires_grad)
                  a->grad = Tensor::full_like(a, 1.0f / (float)log(2))
                                ->div(a)
                                ->mul(out->grad);
              });

  // Trigometric functions
  REGISTER_OP(SIN, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                result = inputs[1];
              }),
              ({
                result->node->inputs = {a};
                result->node->outputs = {result};
                mps->sin(a, result);
              }),
              {
                a = node->inputs[0];
                out = node->outputs[0];
                assert(node->inputs.size() == 1 && node->outputs.size() == 1);
                if (a->requires_grad)
                  a->grad = a->cos()->mul(out->grad);
              });

  REGISTER_OP(COS, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                result = inputs[1];
              }),
              ({
                result->node->inputs = {a};
                result->node->outputs = {result};
                mps->cos(a, result);
              }),
              {
                a = node->inputs[0];
                out = node->outputs[0];
                assert(node->inputs.size() == 1 && node->outputs.size() == 1);
                if (a->requires_grad)
                  a->grad = a->sin()->negate()->mul(out->grad);
              });
  REGISTER_OP(TAN, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                result = inputs[1];
              }),
              ({
                result->node->inputs = {a};
                result->node->outputs = {result};
                mps->tan(a, result);
              }),
              {
                a = node->inputs[0];
                out = node->outputs[0];
                assert(node->inputs.size() == 1 && node->outputs.size() == 1);
                if (a->requires_grad)
                  a->grad = a->cos()->pow(-2.0f)->mul(out->grad);
              });
  REGISTER_OP(
      ASIN, MPS, ({
        assert(inputs.size() == 2);
        a = inputs[0];
        result = inputs[1];
      }),
      ({
        result->node->inputs = {a};
        result->node->outputs = {result};
        mps->asin(a, result);
      }),
      {
        a = node->inputs[0];
        out = node->outputs[0];
        assert(node->inputs.size() == 1 && node->outputs.size() == 1);
        if (a->requires_grad) {
          Tensor *ones = Tensor::ones_like(a);
          a->grad = ones->div(ones->sub(a->pow(2.0f))->sqrt())->mul(out->grad);
        }
      });
  REGISTER_OP(ACOS, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                result = inputs[1];
              }),
              ({
                result->node->inputs = {a};
                result->node->outputs = {result};
                mps->acos(a, result);
              }),
              {
                a = node->inputs[0];
                out = node->outputs[0];
                assert(node->inputs.size() == 1 && node->outputs.size() == 1);
                if (a->requires_grad) {
                  Tensor *ones = Tensor::ones_like(a);
                  a->grad = ones->div(ones->sub(a->pow(2.0f))->sqrt())
                                ->negate()
                                ->mul(out->grad);
                }
              });
  REGISTER_OP(ATAN, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                result = inputs[1];
              }),
              ({
                result->node->inputs = {a};
                result->node->outputs = {result};
                mps->atan(a, result);
              }),
              {
                a = node->inputs[0];
                out = node->outputs[0];
                assert(node->inputs.size() == 1 && node->outputs.size() == 1);
                if (a->requires_grad) {
                  Tensor *ones = Tensor::ones_like(a);
                  a->grad = ones->div(ones->add(a->pow(2.0f)))->mul(out->grad);
                }
              });
  REGISTER_OP(ATAN2, MPS, ({
                assert(inputs.size() == 3);
                a = inputs[0];
                b = inputs[1];
                result = inputs[2];
              }),
              ({
                result->node->inputs = {a, b};
                result->node->outputs = {result};
                mps->atan2(a, b, result);
              }),
              {
                a = node->inputs[0];
                b = node->inputs[1];
                out = node->outputs[0];
                assert(node->inputs.size() == 2 && node->outputs.size() == 1);
                Tensor *denominator = nullptr;
                if (a->requires_grad || b->requires_grad) {
                  denominator = a->pow(2.0f)->add(b->pow(2.0f));
                }
                if (a->requires_grad) {
                  a->grad = b->div(denominator)->negate()->mul(out->grad);
                }
                if (b->requires_grad) {
                  b->grad = a->div(denominator)->mul(out->grad);
                }
              });

  // Hyperbolic functions
  REGISTER_OP(SINH, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                result = inputs[1];
              }),
              ({
                result->node->inputs = {a};
                result->node->outputs = {result};
                mps->sinh(a, result);
              }),
              {
                a = node->inputs[0];
                out = node->outputs[0];
                assert(node->inputs.size() == 1 && node->outputs.size() == 1);
                if (a->requires_grad)
                  a->grad = a->cosh()->mul(out->grad);
              });

  REGISTER_OP(COSH, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                result = inputs[1];
              }),
              ({
                result->node->inputs = {a};
                result->node->outputs = {result};
                mps->cosh(a, result);
              }),
              {
                a = node->inputs[0];
                out = node->outputs[0];
                assert(node->inputs.size() == 1 && node->outputs.size() == 1);
                if (a->requires_grad)
                  a->grad = a->sinh()->mul(out->grad);
              });
  REGISTER_OP(TANH, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                result = inputs[1];
              }),
              ({
                result->node->inputs = {a};
                result->node->outputs = {result};
                mps->tanh(a, result);
              }),
              {
                a = node->inputs[0];
                out = node->outputs[0];
                assert(node->inputs.size() == 1 && node->outputs.size() == 1);
                if (a->requires_grad)
                  a->grad = a->cosh()->pow(-2.0f)->mul(out->grad);
              });
  REGISTER_OP(
      ASINH, MPS, ({
        assert(inputs.size() == 2);
        a = inputs[0];
        result = inputs[1];
      }),
      ({
        result->node->inputs = {a};
        result->node->outputs = {result};
        mps->asinh(a, result);
      }),
      {
        a = node->inputs[0];
        out = node->outputs[0];
        assert(node->inputs.size() == 1 && node->outputs.size() == 1);
        if (a->requires_grad) {
          Tensor *ones = Tensor::ones_like(a);
          a->grad = ones->div(a->pow(2.0f)->add(ones)->sqrt())->mul(out->grad);
        }
      });
  REGISTER_OP(
      ACOSH, MPS, ({
        assert(inputs.size() == 2);
        a = inputs[0];
        result = inputs[1];
      }),
      ({
        result->node->inputs = {a};
        result->node->outputs = {result};
        mps->acosh(a, result);
      }),
      {
        a = node->inputs[0];
        out = node->outputs[0];
        assert(node->inputs.size() == 1 && node->outputs.size() == 1);
        if (a->requires_grad) {
          Tensor *ones = Tensor::ones_like(a);
          a->grad = ones->div(a->pow(2.0f)->sub(ones)->sqrt())->mul(out->grad);
        }
      });
  REGISTER_OP(ATANH, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                result = inputs[1];
              }),
              ({
                result->node->inputs = {a};
                result->node->outputs = {result};
                mps->atanh(a, result);
              }),
              {
                a = node->inputs[0];
                out = node->outputs[0];
                assert(node->inputs.size() == 1 && node->outputs.size() == 1);
                if (a->requires_grad) {
                  Tensor *ones = Tensor::ones_like(a);
                  a->grad = ones->div(ones->sub(a->pow(2.0f)))->mul(out->grad);
                }
              });
  // initalisations;
  REGISTER_OP(ONES_INIT, MPS, ({
                assert(inputs.size() == 1);
                a = inputs[0];
              }),
              ({ mps->ones(a); }), {});

  REGISTER_OP(ZEROES_INIT, MPS, ({
                assert(inputs.size() == 1);
                a = inputs[0];
              }),
              ({ mps->zeros(a); }), {});
  REGISTER_OP(EYE_INIT, MPS, ({
                assert(inputs.size() == 1);
                a = inputs[0];
              }),
              ({ mps->eye(a); }), {});

  REGISTER_OP(FULL_INIT, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                b = inputs[1];
              }),
              ({ mps->full(a, b); }), {});
  REGISTER_OP(CLONE, MPS, ({
                throw std::logic_error(
                    "method not supposed to be called through dispatcher");
              }),
              ({}), {
                a = node->inputs[0];
                out = node->outputs[0];
                if (a->requires_grad) {
                  a->grad = Tensor::clone(out->grad);
                }
              });
}
