#pragma once

#include "Runtime.h"

using namespace mkldnn;

namespace torch { namespace mkldnn {

struct Context {
  Context(std::vector<int> pad, std::vector<int> stride)
    : pad(memory::dims({pad[0], pad[1]}))
    , stride(memory::dims({stride[0], stride[1]}))
    {}

  memory::dims pad;
  memory::dims stride;

  std::shared_ptr<convolution_forward::primitive_desc> conv_forward_pd;
  std::shared_ptr<convolution_backward_data::primitive_desc> conv_backward_data_pd;
  std::shared_ptr<convolution_backward_weights::primitive_desc> conv_backward_weight_pd;
};

Context* mkldnn_convolution_forward(
  at::Tensor& input,
  at::Tensor& output,
  at::Tensor& weight,
  at::Tensor& bias,
  std::vector<int> pad,
  std::vector<int> stride);

void mkldnn_convolution_backward_data(
  at::Tensor& grad_output,
  at::Tensor& weight,
  at::Tensor& grad_input,
  Context* ctx);

void mkldnn_convolution_backward_weight(
  at::Tensor& input,
  at::Tensor& grad_output,
  at::Tensor& grad_weight,
  at::Tensor& grad_bias,
  Context* ctx);

}}  // namespace torch::mkldnn
