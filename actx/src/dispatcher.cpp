#include "dispatcher.h"
#include "device_type.h"
#include "main.h"
#include "op_types.h"
#include "opnode.h"
#include "tensor.h"
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

void Dispatcher::init_register() {
  REGISTER_OP(NEGATE, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                b = inputs[1];
              }),
              ({ mps->negate(a, b); }), {});

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
                a->grad = out->grad;
                b->grad = out->grad;
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
                a->grad = out->grad;
                b->grad = out->grad->negate(false);
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
                a->grad = b->mul(out->grad, false);
                b->grad = a->mul(out->grad, false);
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
                a->grad = out->grad->div(b, false);
                b->grad = a->div(b->pow(2.0f, false), false)
                              ->mul(out->grad, false)
                              ->negate(false);
              }));
  REGISTER_OP(POW, MPS, ({
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
                // BUG: wrong backward method
                assert(node->inputs.size() == 2 && node->outputs.size() == 1);
                a = node->inputs[0];
                b = node->inputs[1];
                out = node->outputs[0];
                a->grad = out->grad->div(b, false);
                b->grad =
                    a->div(b->pow(2.0f, false), false)->mul(out->grad, false);
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
              {});

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
              {});

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
              {});

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
              {});

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
              {});

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
              {});

  // math functions;
  REGISTER_OP(SQRT, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                b = inputs[1];
              }),
              ({ mps->sqrt(a, b); }), {});

  REGISTER_OP(EXP, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                b = inputs[1];
              }),
              ({ mps->exp(a, b); }), {});

  REGISTER_OP(LOG, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                b = inputs[1];
              }),
              ({ mps->log(a, b); }), {});
  REGISTER_OP(LOG10, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                b = inputs[1];
              }),
              ({ mps->log10(a, b); }), {});

  REGISTER_OP(LOG2, MPS, ({
                assert(inputs.size() == 2);
                a = inputs[0];
                b = inputs[1];
              }),
              ({ mps->log2(a, b); }), {});

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
}
