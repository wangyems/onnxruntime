// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/data_types_internal.h"
#include "core/framework/op_kernel.h"

#include "Featurizers/LagLeadOperatorFeaturizer.h"
#include "Featurizers/../Archive.h"

namespace onnxruntime {
namespace featurizers {

template <typename TargetT>
struct LagLeadOperatorTransformerImpl {
  void operator()(OpKernelContext* ctx) const {
    // Create the transformer
    Microsoft::Featurizer::Featurizers::LagLeadOperatorTransformer<TargetT> transformer(
        [ctx](void) {
          const auto* state_tensor(ctx->Input<Tensor>(0));
          const uint8_t* const state_data(state_tensor->Data<uint8_t>());

          Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
          return Microsoft::Featurizer::Featurizers::LagLeadOperatorTransformer<TargetT>(archive);
        }());

    // Get the Grains
    const auto* grains_tensor(ctx->Input<Tensor>(1));
    const std::string* grains_data(input_tensor->Data<std::string>());
    const auto grains_num = grains_tensor.Shape()[1];

    // Get the Target
    const auto* target_tensor(ctx->Input<Tensor>(2));
    const TargetT* target_data(input_tensor->Data<TargetT>());

    // Prepare the output
    const auto output_dim_0 = grains_tensor.Shape()[0];
    const auto output_dim_1 = transformer.getRowsNumber();
    const auto output_dim_2 = transformer.getColsNumber();
    TensorShape output_shape({output_dim_0, output_dim_1, output_dim_2});
    Tensor* output_tensor(ctx->Output(0, output_shape));
    double* output_data(output_tensor->MutableData<double>());

    // Transform
    std::vector<std::string> grains;
    grains.reserve(grains_num);
    for (int64_t i = 0; i < output_dim_0; ++i) {
      //Prepare Input and Output
      grains.clear();
      std::copy(grains_data, grains_data + grains_num, std::back_inserter(grains));
      std::tuple<std::vector<std::string>, TargetT> input_per_row = std::make_tuple(std::move(grains), *target_data);
      Eigen::Map<NS::RowMajMatrix<double>> output_per_row(output_data, output_dim_1, output_dim_2);

      //Execute
      output_per_row = transformer.execute(input_per_row);

      target_data++;
      grains_data += grains_num;
      output_data += output_matrix_per_row.size();
    }
  }
};

class LagLeadOperatorTransformer final : public OpKernel {
 public:
  explicit LagLeadOperatorTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    utils::MLTypeCallDispatcher<LagLeadOperatorTransformerImpl, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
                                int64_t, uint64_t, float, double>
        t_disp(ctx->Input<Tensor>(1)->GetElementType());
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
        .TypeConstraint("TargetT", {DataTypeImpl::GetTensorType<int8_t>(),
                                   DataTypeImpl::GetTensorType<uint8_t>(),
                                   DataTypeImpl::GetTensorType<int16_t>(),
                                   DataTypeImpl::GetTensorType<uint16_t>(),
                                   DataTypeImpl::GetTensorType<int32_t>(),
                                   DataTypeImpl::GetTensorType<uint32_t>(),
                                   DataTypeImpl::GetTensorType<int64_t>(),
                                   DataTypeImpl::GetTensorType<uint64_t>(),
                                   DataTypeImpl::GetTensorType<float>(),
                                   DataTypeImpl::GetTensorType<double>()
                                   }),
    LagLeadOperatorTransformer);
}  // namespace featurizers
}  // namespace onnxruntime
