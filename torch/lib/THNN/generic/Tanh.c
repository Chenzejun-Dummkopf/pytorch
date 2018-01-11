#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Tanh.c"
#else

void THNN_(Tanh_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output)
{
  THTensor_(tanh)(output, input);
}

void THNN_(Tanh_updateGradInput)(
          THNNState *state,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output)
{
  THNN_CHECK_SHAPE(output, gradOutput);
  THTensor_(resizeAs)(gradInput, output);

  int64_t gradInputSize = THTensor_(nElement)(gradInput);
  int64_t gradOutputSize = THTensor_(nElement)(gradOutput);
  int64_t outputSize = THTensor_(nElement)(output);
  int gradInputContig = THTensor_(isContiguous)(gradInput)? 1:0;
  int gradOutputContig = THTensor_(isContiguous)(gradOutput)? 1:0;
  int outputContig = THTensor_(isContiguous)(output)? 1:0;
  int serial_path = 0;

  if((gradInputSize = gradOutputSize) && (gradInputSize == outputSize) ){
    if(gradInputContig && gradOutputContig && outputContig) {
      real* ptr_gradOutput = THTensor_(data)(gradOutput);
      real* ptr_gradInput  = THTensor_(data)(gradInput);
      real* ptr_output     = THTensor_(data)(output);
      long i;
#if _OPENMP
      #pragma omp parallel for private(i)
#endif
      for (i = 0; i < outputSize; i++)
      {
        real z = ptr_output[i];
        ptr_gradInput[i] = ptr_gradOutput[i] * (1. - z*z);
      }
    } else {
#if _OPENMP
      int inOMP = omp_in_parallel();
      if (inOMP){
        serial_path = 1;
      } else {
        TH_TENSOR_APPLY3_OMP(outputSize, gradInputContig, gradOutputContig, outputContig,
                                         real, gradInput, real, gradOutput, real, output,
                                         real z = *output_data;
                                         *gradInput_data = *gradOutput_data * (1. - z*z);
        );
      }
#else
      serial_path = 1;
#endif
    }
  } else {
    serial_path = 1;
  }
  if(serial_path){
    TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,
      real z = *output_data;
      *gradInput_data = *gradOutput_data * (1. - z*z);
    );
  }

}

#endif
