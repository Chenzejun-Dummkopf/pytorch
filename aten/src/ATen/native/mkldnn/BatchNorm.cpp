#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#if !AT_MKLDNN_ENABLED()

namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& running_mean, const at::Tensor& running_var,
    bool training, double momentum, double eps){
  throw std::runtime_error("mkldnn_batch_norm_forward: ATen not compiled with MKLDNN support");
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& running_mean, const at::Tensor& running_var,
    const at::Tensor& save_mean, const at::Tensor& save_var, double eps) {
  throw std::runtime_error("mklnn_batch_norm_backward: ATen not compiled with MKLDNN support");
}

}}

#else // AT_MKLDNN_EBABLED

#include <ATen/mkldnn/Runtime.h>

using namespace mkldnn;

namespace at { namespace native {

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& running_mean, const at::Tensor& running_var,
    bool training, double momentum, double eps)
{
  // weight bias? use or not
  // test, use running mean and bias
  // test, use batch info
  // train, update running stats, use batch info
  // train, do not update running stats, use batch info
  unsigned flags = 0;
  auto propagation = training ? prop_kind::forward_training : prop_kind::forward_inference;
  bool use_weight_bias_  = (weight.defined() && bias.defined()) ? true : false;
  bool use_running_stat = (running_mean.defined() && running_var.defined()) ? true : false;
  if (use_weight_bias_)  flags |= use_scale_shift;
  if (use_running_stat && (!training) ) flags |= use_global_stats;

  int32_t n = input.size(0);
  int32_t ic = input.size(1);
  int32_t ih = input.size(2);
  int32_t iw = input.size(3);

  auto cpu_engine = CpuEngine::Instance().get_engine();

  //only support f32
  memory::data_type my_data_type = memory::data_type::f32;

  //always create output for pytorch only allow in-place
  auto output = input.type().tensor(input.sizes());

  //user input
  auto input_usr_memory = memory({{{n, ic, ih, iw}, my_data_type, memory::format::nchw}, cpu_engine},
    input.data_ptr());
  auto output_usr_memory = memory({{{n, ic, ih, iw}, my_data_type, memory::format::nchw}, cpu_engine},
    output.data_ptr());

  // ---- Initialize BatchNorm primitive descriptor -------------
  //TODO when to use saved mean? train or batch=1?
  //TODO float double type control?
  auto input_md  = memory::desc({n, ic, ih, iw}, my_data_type, memory::format::nchw);
  std::shared_ptr<batch_normalization_forward::desc> bn_forward_desc;
  std::shared_ptr<batch_normalization_forward::primitive_desc> bn_forward_pd;
  bn_forward_desc.reset(new batch_normalization_forward::desc(propagation, input_md, eps, flags));
  bn_forward_pd.reset(new batch_normalization_forward::primitive_desc(*bn_forward_desc, cpu_engine));

  // ---- for scaleshift ---------------------
  std::shared_ptr<memory> scaleshift_memory;
  if (use_weight_bias_) {
    scaleshift_memory.reset(new memory(bn_forward_pd->weights_primitive_desc()));
    float* scaleshift_buf = reinterpret_cast<float *>(scaleshift_memory->get_data_handle());
    for (int32_t i = 0; i < ic; ++i) {
      scaleshift_buf[i] = ((float*)weight.data_ptr())[i];   // weight
    }
    for (int32_t i = 0; i < ic; ++i) {
      scaleshift_buf[ic + i] = ((float*)bias.data_ptr())[i];  // bias
    }
  }

  //set output memory
  auto output_pd = bn_forward_pd->dst_primitive_desc();
  auto output_memory = output_usr_memory;
  if (output_usr_memory.get_primitive_desc() != memory::primitive_desc(output_pd)) {
    output_memory = memory(output_pd);
  }

  //construct bn op
  //TODO memory release?
  std::shared_ptr<batch_normalization_forward> bn_forward;
  std::shared_ptr<memory> mean_memory, variance_memory;
  Tensor save_mean = input.type().tensor();
  Tensor save_var = input.type().tensor();
  save_mean.resize_({ic});
  save_var.resize_({ic});
 
  if ( !training ) {
    //test
    if (use_running_stat){
      //use global data, set mean and variance
      mean_memory.reset(new memory(bn_forward_pd->mean_primitive_desc(), running_mean.data_ptr()));
      variance_memory.reset(new memory(bn_forward_pd->variance_primitive_desc(), running_var.data_ptr()));
      if (use_weight_bias_){
        bn_forward.reset(new batch_normalization_forward(*bn_forward_pd, input_usr_memory,
          mkldnn::primitive::at(*mean_memory), mkldnn::primitive::at(*variance_memory),
          *scaleshift_memory, output_memory));
      }
      else{
        bn_forward.reset(new batch_normalization_forward(*bn_forward_pd, input_usr_memory,
          mkldnn::primitive::at(*mean_memory), mkldnn::primitive::at(*variance_memory), output_memory));
      }
    }
    else {
      if (use_weight_bias_){
        bn_forward.reset(new batch_normalization_forward(*bn_forward_pd, input_usr_memory,
          *scaleshift_memory, output_memory));
      }
      else{
        bn_forward.reset(new batch_normalization_forward(*bn_forward_pd, input_usr_memory,
          output_memory));
      }
    }
  }
  else {
    //calculate mean and variance and stored in mean and variance
    mean_memory.reset(new memory(bn_forward_pd->mean_primitive_desc(), save_mean.data_ptr()));
    variance_memory.reset(new memory(bn_forward_pd->variance_primitive_desc(), save_var.data_ptr()));
    if (use_weight_bias_) {
      bn_forward.reset(new batch_normalization_forward(*bn_forward_pd, input_usr_memory,
        *scaleshift_memory, output_memory, *mean_memory, *variance_memory));
    }
    else {
      bn_forward.reset(new batch_normalization_forward(*bn_forward_pd, input_usr_memory,
        output_memory, *mean_memory, *variance_memory));
    }
  }

  std::vector<primitive> net;
  net.push_back(*bn_forward);

  //reorder for output
  //pytorch do not allow inplace
  //always allocate memory for output
  if (output_usr_memory.get_primitive_desc() != memory::primitive_desc(output_pd)) {
    net.push_back(reorder(output_memory, output_usr_memory));
  }

  //execute
  Stream::Instance().get_stream().submit(net);
  if (training && use_running_stat) {
    const float len = (float)(n * ih * iw);
    float* mean_buf = reinterpret_cast<float *>(mean_memory->get_data_handle());
    float* var_buf = reinterpret_cast<float *>(variance_memory->get_data_handle());
    float* running_mean_buf = reinterpret_cast<float *>(running_mean.data_ptr());
    float* running_var_buf = reinterpret_cast<float *>(running_var.data_ptr());
    const float reborn = 1.0f - momentum;
    const float adjust = momentum * len / (len - 1);
    for (int32_t i=0; i<ic; ++i){
        running_mean_buf[i] = running_mean_buf[i] * reborn + mean_buf[i] * momentum;
        running_var_buf[i]  = running_var_buf[i] * reborn + var_buf[i] * adjust;
    }
  }
  return std::tuple<Tensor, Tensor, Tensor>{output, save_mean, save_var};
}

std::tuple<Tensor, Tensor, Tensor> mkldnn_batch_norm_backward(
    const at::Tensor& input, const at::Tensor& grad_output,
    const at::Tensor& weight, const at::Tensor& bias,
    const at::Tensor& running_mean, const at::Tensor& running_var,
    const at::Tensor& save_mean, const at::Tensor& save_var, double eps)  {

  unsigned flags = 0;
  bool use_weight_bias_  = (weight.defined() && bias.defined()) ? true : false;
  bool use_running_stat = (running_mean.defined() && running_var.defined()) ? true : false;
  if (use_weight_bias_)  flags |= use_scale_shift;

  int32_t n = input.size(0);
  int32_t ic = input.size(1);
  int32_t ih = input.size(2);
  int32_t iw = input.size(3);

  auto cpu_engine = CpuEngine::Instance().get_engine();

  //only support f32
  memory::data_type my_data_type = memory::data_type::f32;

  //always create output for pytorch only allow in-place
  auto grad_input  = input.type().tensor(input.sizes());
  //TODO what's the type? undefined?
  auto grad_weight = input.type().tensor();
  auto grad_bias = input.type().tensor();
  if (use_weight_bias_) {
    grad_weight.resize_(weight.sizes());
    grad_bias.resize_(weight.sizes());
  }

  //input and weight
  auto input_usr_memory = memory({{{n, ic, ih, iw}, my_data_type, memory::format::nchw}, cpu_engine},
    input.data_ptr());

  //user grad
  auto grad_output_usr_memory = memory({{{n, ic, ih, iw}, my_data_type, memory::format::nchw}, cpu_engine},
    grad_output.data_ptr());
  auto grad_input_usr_memory = memory({{{n, ic, ih, iw}, my_data_type, memory::format::nchw}, cpu_engine},
    grad_input.data_ptr());

  // ---- Initialize BatchNorm primitive descriptor -------------
  auto input_md  = memory::desc({n, ic, ih, iw}, my_data_type, memory::format::nchw);
  std::shared_ptr<batch_normalization_forward::desc> bn_forward_desc;
  std::shared_ptr<batch_normalization_forward::primitive_desc> bn_forward_pd;
  bn_forward_desc.reset(new batch_normalization_forward::desc(prop_kind::forward_training, input_md, eps, flags));
  bn_forward_pd.reset(new batch_normalization_forward::primitive_desc(*bn_forward_desc, cpu_engine));

  //TODO create?? something wrong
  //auto grad_output_pd = bn_forward_pd->dst_primitive_desc();
  auto grad_output_md = input_md;
  std::shared_ptr<batch_normalization_backward::desc> bn_backward_desc;
  std::shared_ptr<batch_normalization_backward::primitive_desc> bn_backward_pd;
  bn_backward_desc.reset(new batch_normalization_backward::desc(prop_kind::backward, input_md, grad_output_md, eps, flags));
  bn_backward_pd.reset(new batch_normalization_backward::primitive_desc(*bn_backward_desc, cpu_engine, *bn_forward_pd));

  std::vector<primitive> net;

  //set grad output memory
  auto grad_output_memory = grad_output_usr_memory;
  //if (grad_output_memory.get_primitive_desc() != grad_output_pd) {
  //  grad_output_memory = memory(grad_output_pd);
  //  net.push_back(reorder(grad_output_usr_memory, grad_output_memory));
  //}

  //set grad input memory
  auto grad_input_memory = grad_input_usr_memory;

  //grad weight and bias
  std::shared_ptr<memory> grad_scaleshift_memory;
  grad_scaleshift_memory.reset(new memory(bn_backward_pd->diff_weights_primitive_desc()));

  // ---- for scaleshift ---------------------
  std::shared_ptr<memory> scaleshift_memory;
  scaleshift_memory.reset(new memory(bn_forward_pd->weights_primitive_desc()));
  if (use_weight_bias_) {
    float* scaleshift_buf = reinterpret_cast<float *>(scaleshift_memory->get_data_handle());
    for (int32_t i = 0; i < ic; ++i) {
      scaleshift_buf[i] = ((float*)weight.data_ptr())[i];   // weight
    }
    for (int32_t i = 0; i < ic; ++i) {
      scaleshift_buf[ic + i] = ((float*)bias.data_ptr())[i];  // bias
    }
  }
  else {
    float* scaleshift_buf = reinterpret_cast<float *>(scaleshift_memory->get_data_handle());
    for (int32_t i = 0; i < ic; ++i) {
      scaleshift_buf[i] = 0.0;   // weight
    }
    for (int32_t i = 0; i < ic; ++i) {
      scaleshift_buf[ic + i] = 1.0;  // bias
    }

  }
  //construct bn op
  auto mean_memory = memory(bn_backward_pd->mean_primitive_desc(), save_mean.data_ptr());
  auto variance_memory = memory(bn_backward_pd->variance_primitive_desc(), save_var.data_ptr());
  std::shared_ptr<batch_normalization_backward> bn_backward;
  if (use_weight_bias_) {
    bn_backward.reset(new batch_normalization_backward(*bn_backward_pd, input_usr_memory,
      mkldnn::primitive::at(mean_memory), mkldnn::primitive::at(variance_memory),
      grad_output_memory, *scaleshift_memory,
      grad_input_memory, *grad_scaleshift_memory));

  }
  else {
    bn_backward.reset(new batch_normalization_backward(*bn_backward_pd, input_usr_memory,
      mkldnn::primitive::at(mean_memory), mkldnn::primitive::at(variance_memory),
      grad_output_memory, *scaleshift_memory,
      grad_input_memory, *grad_scaleshift_memory));
  }
  net.push_back(*bn_backward);

  //execute
  Stream::Instance().get_stream().submit(net);

  // ---- for scaleshift ---------------------
  if (use_weight_bias_) {
    float* grad_scaleshift_buf = reinterpret_cast<float *>(grad_scaleshift_memory->get_data_handle());
    for (int32_t i = 0; i < ic; ++i) {
      ((float*)grad_weight.data_ptr())[i] = grad_scaleshift_buf[i];   // weight
    }
    for (int32_t i = 0; i < ic; ++i) {
      ((float*)grad_bias.data_ptr())[i] = grad_scaleshift_buf[ic + i];  // bias
    }
  }
  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}
}}  // namespace at::native
#endif
