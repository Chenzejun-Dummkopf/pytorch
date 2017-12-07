#include "Runtime.h"

using namespace mkldnn;

namespace torch { namespace mkldnn {

#ifdef MKLDNN_DEBUG
void print_tensor(at::Tensor& tensor, int n, std::string s) {
  std::cout << s << "\t";
  auto tensor_p = (float*)tensor.data_ptr();
  for (int i = 0; i < n; i++) {
      std::cout << tensor_p[i] << " ";
  }
  std::cout << std::endl;
}

void tensor_compare(at::Tensor& left, at::Tensor& right, std::string s) {
  float* left_p = (float*)left.data_ptr();
  float* right_p = (float*)right.data_ptr();
  float sum_l = 0;
  float sum_r = 0;

  for (int64_t i = 0; i < left.numel(); i++) {
    sum_l += left_p[i];
    sum_r += right_p[i];
  }

  std::cout << s << " accumulated error " << (sum_l - sum_r) / sum_r << std::endl;
}
#endif

}}  // namespace torch::mkldnn
