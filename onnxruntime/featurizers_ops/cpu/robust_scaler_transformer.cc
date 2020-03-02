// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/RobustScalerFeaturizer.h"
#include "Featurizers/../Archive.h"

namespace onnxruntime {
namespace featurizers {

template <typename T>
struct OutputTypeMapper {};
template <>
struct OutputTypeMapper<int8_t> { using type = float; };
template <>
struct OutputTypeMapper<int16_t> { using type = float; };
template <>
struct OutputTypeMapper<uint8_t> { using type = float; };
template <>
struct OutputTypeMapper<uint16_t> { using type = float; };
template <>
struct OutputTypeMapper<float> { using type = float; };
template <>
struct OutputTypeMapper<int32_t> { using type = double; };
template <>
struct OutputTypeMapper<int64_t> { using type = double; };
template <>
struct OutputTypeMapper<uint32_t> { using type = double; };
template <>
struct OutputTypeMapper<uint64_t> { using type = double; };
template <>
struct OutputTypeMapper<double> { using type = double; };

template <typename InputT>
struct RobustScalerTransformerImpl {
  void operator()(OpKernelContext* ctx) const {
    // Create the transformer
    Microsoft::Featurizer::Featurizers::RobustScalerTransformer<InputT, typename OutputTypeMapper<InputT>::type> transformer(
        [ctx](void) {
          const auto* state_tensor(ctx->Input<Tensor>(0));
          const uint8_t* const state_data(state_tensor->Data<uint8_t>());

          Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
          return Microsoft::Featurizer::Featurizers::RobustScalerTransformer<InputT, typename OutputTypeMapper<InputT>::type>(archive);
        }());

    // Get the input
    const auto* input_tensor(ctx->Input<Tensor>(1));
    const InputT* input_data(input_tensor->Data<InputT>());

    // Prepare the output
    Tensor* output_tensor(ctx->Output(0, input_tensor->Shape()));
    typename OutputTypeMapper<InputT>::type* output_data(output_tensor->MutableData<typename OutputTypeMapper<InputT>::type>());

    // Execute
    const int64_t length(input_tensor->Shape().Size());

    for (int64_t i = 0; i < length; ++i) {
      output_data[i] = transformer.execute(input_data[i]);
    }
  }
};

class RobustScalerTransformer final : public OpKernel {
 public:
  explicit RobustScalerTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    utils::MLTypeCallDispatcher<RobustScalerTransformerImpl,
                                double>
        t_disp(ctx->Input<Tensor>(1)->GetElementType());
    t_disp.Invoke(ctx);
    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    RobustScalerTransformer,
    kMSFeaturizersDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T0", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("InputT", {

                                   DataTypeImpl::GetTensorType<double>()}),
    RobustScalerTransformer);

}  // namespace featurizers
}  // namespace onnxruntime
