#pragma once

#include <stdlib.h>
#include <mkldnn.hpp>
#include <ATen/ATen.h>

using namespace mkldnn;

namespace torch { namespace mkldnn {

#ifdef MKLDNN_DEBUG
void print_tensor(at::Tensor& tensor, int n, std::string s);
void tensor_compare(at::Tensor& left, at::Tensor& right, std::string s);
#endif

// CpuEngine singleton
struct CpuEngine {
  static CpuEngine& Instance() {
    static CpuEngine myInstance;
    return myInstance;
  }
  engine& get_engine() {
    return _cpu_engine;
  }
  CpuEngine(CpuEngine const&) = delete;
  CpuEngine& operator=(CpuEngine const&) = delete;

protected:
  CpuEngine():_cpu_engine(engine::cpu, 0) {}
  ~CpuEngine() {}

private:
  engine _cpu_engine;
};

// Stream singleton
struct Stream {
  static Stream& Instance() {
    static Stream myInstance;
    return myInstance;
  };
  stream& get_stream() {
    return _cpu_stream;
  }
  Stream(Stream const&) = delete;
  Stream& operator=(Stream const&) = delete;

protected:
  Stream():_cpu_stream(stream::kind::eager) {}
  ~Stream() {}

private:
  stream _cpu_stream;
};


}}  // namespace torch::mkldnn
