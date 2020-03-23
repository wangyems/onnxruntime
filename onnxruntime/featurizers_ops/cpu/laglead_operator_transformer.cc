// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/LagLeadOperatorFeaturizer.h"
#include "Featurizers/../Archive.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace featurizers {

template <typename T>
struct LagLeadOperatorTransformerImpl {
  void operator()(OpKernelContext* ctx) const {
    using OutputType = NS::RowMajMatrix<nonstd::optional<T>>;
    // Create the transformer
    Microsoft::Featurizer::Featurizers::LagLeadOperatorTransformer<T> transformer(
        [ctx](void) {
          const auto* state_tensor(ctx->Input<Tensor>(0));
          const uint8_t* const state_data(state_tensor->Data<uint8_t>());

          Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
          return Microsoft::Featurizer::Featurizers::LagLeadOperatorTransformer<T>(archive);
        }());

    // Get the Grains
    const auto* grains_tensor(ctx->Input<Tensor>(1));
    //const std::string* grains_data(grains_tensor->Data<std::string>());
    //const int64_t grains_num = grains_tensor->Shape()[1];

    // Get the Target
    const auto* target_tensor(ctx->Input<Tensor>(2));
    const T* target_data(target_tensor->Data<T>());

    // Prepare the output
    const int64_t output_dim_0 = grains_tensor->Shape()[0];
    //const int64_t  output_dim_1 = transformer.getRowsNumber();
    //const int64_t  output_dim_2 = transformer.getColsNumber();
    //hard code for temporary
    const int64_t output_dim_1 = 2;
    const int64_t output_dim_2 = 1;
    TensorShape output_shape({output_dim_0, output_dim_1, output_dim_2});
    Tensor* output_tensor(ctx->Output(0, output_shape));
    T* output_data(output_tensor->MutableData<T>());

    std::function<void(OutputType)> callback_fn;
    callback_fn = [&output_data, &output_dim_1, &output_dim_2](OutputType value) -> void {
      for (int32_t i = 0; i < output_dim_1; ++i) {
        for (int32_t j = 0; j < output_dim_2; ++j) {
          if (value(i, j).has_value()) {
            *output_data++ = *value(i, j);
          } else {
            *output_data++ = NS::Traits<T>::CreateNullValue();
            //*output_data++ = static_cast<T>(0);
          }
        }
      }
      //OutputType output_per_row(output_data, output_dim_1, output_dim_2);
      //output_per_row = value;
      //output_data += output_per_row.size();
    };

    // Transform
    //std::vector<std::string> grains;
    //grains.reserve(grains_num);
    for (int64_t i = 0; i < output_dim_0; ++i) {
      //Prepare Input and Output
      //grains.clear();
      //std::copy(grains_data, grains_data + grains_num, std::back_inserter(grains));
      //std::tuple<std::vector<std::string>, TargetT> input_per_row = std::make_tuple(std::move(grains), *target_data);

      //Execute
      transformer.execute(*target_data++, callback_fn);

      //grains_data += grains_num;
    }
    transformer.flush(callback_fn);
  }
};

class LagLeadOperatorTransformer final : public OpKernel {
 public:
  explicit LagLeadOperatorTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    utils::MLTypeCallDispatcher<LagLeadOperatorTransformerImpl, //int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
                                //int64_t, uint64_t,
                                float, double>//, std::string>
        t_disp(ctx->Input<Tensor>(2)->GetElementType());
    t_disp.Invoke(ctx);
    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    LagLeadOperatorTransformer,
    kMSFeaturizersDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T0", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("GrainT", DataTypeImpl::GetTensorType<std::string>())
        .TypeConstraint("T", {//DataTypeImpl::GetTensorType<int8_t>(),
                              //DataTypeImpl::GetTensorType<uint8_t>(),
                              //DataTypeImpl::GetTensorType<int16_t>(),
                              //DataTypeImpl::GetTensorType<uint16_t>(),
                              //DataTypeImpl::GetTensorType<int32_t>(),
                              //DataTypeImpl::GetTensorType<uint32_t>(),
                              //DataTypeImpl::GetTensorType<int64_t>(),
                              //DataTypeImpl::GetTensorType<uint64_t>(),
                              DataTypeImpl::GetTensorType<float>(),
                              DataTypeImpl::GetTensorType<double>()
                              //DataTypeImpl::GetTensorType<std::string>()
                              }),
    LagLeadOperatorTransformer);
}  // namespace featurizers
}  // namespace onnxruntime
