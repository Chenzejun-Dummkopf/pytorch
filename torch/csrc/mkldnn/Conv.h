#pragma once

#include "Runtime.h"

using namespace mkldnn;

namespace torch { namespace mkldnn {

#if 1
struct Context {
  std::shared_ptr<convolution_forward::primitive_desc> conv_forward_pd;
  std::shared_ptr<convolution_backward_data::primitive_desc> conv_backward_data_pd;
  std::shared_ptr<convolution_backward_weights::primitive_desc> conv_backward_weight_pd;
};
#endif

void mkldnn_convolution_forward(
  at::Tensor& input,
  at::Tensor& output,
  at::Tensor& weight,
  at::Tensor& bias,
  std::vector<int> pad,
  std::vector<int> stride);


void mkldnn_convolution_backward(
  at::Tensor& input,
  at::Tensor& grad_output,
  at::Tensor& weight,
  at::Tensor& grad_input,
  at::Tensor& grad_weight,
  at::Tensor& grad_bias,
  std::vector<int> pad,
  std::vector<int> stride);

}}  // namespace torch::mkldnn
