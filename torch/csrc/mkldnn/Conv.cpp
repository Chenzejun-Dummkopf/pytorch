#include "Conv.h"

using namespace mkldnn;

namespace torch { namespace mkldnn {

void mkldnn_convolution_forward(
    at::Tensor& input,
    at::Tensor& output, 
    at::Tensor& weight,
    at::Tensor& bias,
    std::vector<int> pad,
    std::vector<int> stride)
{
  //TODO: mkldnn arg check?
  //dilation?
  auto cpu_engine = CpuEngine::Instance().get_engine();
  
  int32_t n = input.size(0);
  int32_t ic = input.size(1);
  int32_t ih = input.size(2);
  int32_t iw = input.size(3);

  int32_t oc = output.size(1);
  int32_t oh = output.size(2);
  int32_t ow = output.size(3);

  int32_t kh = weight.size(2);
  int32_t kw = weight.size(3);

  memory::dims conv_src_tz = {n, ic, ih, iw};
  memory::dims conv_weights_tz = {oc, ic, kh, kw};
  memory::dims conv_dst_tz = {n, oc, oh, ow};
  memory::dims conv_strides = {stride[0], stride[1]};
  memory::dims conv_padding = {pad[0], pad[1]};

  auto conv_user_src_memory = memory({{{conv_src_tz}, memory::data_type::f32,
    memory::format::nchw}, cpu_engine}, input.data_ptr());
  auto conv_user_weight_memory = memory({{{conv_weights_tz}, memory::data_type::f32, 
    memory::format::oihw}, cpu_engine}, weight.data_ptr());
  auto conv_user_dst_memory = memory({{{conv_dst_tz}, memory::data_type::f32,
    memory::format::nchw}, cpu_engine}, output.data_ptr());

  auto conv_src_md = memory::desc({conv_src_tz}, memory::data_type::f32,
    memory::format::any);
  auto conv_weight_md = memory::desc({conv_weights_tz}, memory::data_type::f32, 
    memory::format::any);
  auto conv_dst_md = memory::desc({conv_dst_tz}, memory::data_type::f32,
    memory::format::any);

  //std::unique_ptr<Context> ctx(new Context());
  memory::dims conv_bias_tz = {oc};
  auto conv_user_bias_memory = memory({{{conv_bias_tz}, memory::data_type::f32,
    memory::format::x}, cpu_engine}, bias.data_ptr());
  auto conv_bias_md = memory::desc({conv_bias_tz}, memory::data_type::f32,
    memory::format::any);
  auto conv_desc = convolution_forward::desc(prop_kind::forward,
    convolution_direct, conv_src_md, conv_weight_md, conv_bias_md,
    conv_dst_md, conv_strides, conv_padding, conv_padding,
    padding_kind::zero);

  auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, cpu_engine);

  std::vector<primitive> net;

  auto conv_src_memory = conv_user_src_memory;
  if (memory::primitive_desc(conv_prim_desc.src_primitive_desc()) !=
    conv_user_src_memory.get_primitive_desc()) {
    conv_src_memory = memory(conv_prim_desc.src_primitive_desc());
    net.push_back(reorder(conv_user_src_memory, conv_src_memory));
  }

  auto conv_weight_memory = conv_user_weight_memory;
  if (memory::primitive_desc(conv_prim_desc.weights_primitive_desc()) !=
    conv_user_weight_memory.get_primitive_desc()) {
    conv_weight_memory = memory(conv_prim_desc.weights_primitive_desc());
    net.push_back(reorder(conv_user_weight_memory, conv_weight_memory));
  }

  auto conv_dst_memory = memory(conv_prim_desc.dst_primitive_desc());

  net.push_back(convolution_forward(conv_prim_desc, conv_src_memory,
    conv_weight_memory, conv_user_bias_memory, conv_dst_memory));

  if (conv_dst_memory != conv_user_dst_memory) {
    net.push_back(reorder(conv_dst_memory, conv_user_dst_memory));
  }

  stream(stream::kind::eager).submit(net).wait();

  //return ctx.release();
}

}}  // namespace torch::mkldnn
