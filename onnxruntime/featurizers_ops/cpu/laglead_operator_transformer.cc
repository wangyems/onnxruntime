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

    using GrainT = std::vector<std::string>;
    using EstimatorT = Microsoft::Featurizer::Featurizers::GrainedLagLeadOperatorEstimator<T>;
    using GrainedInputType = std::tuple<GrainT, T>;
    using OutputMatrixDataType = NS::Traits<T>::nullable_type;
    using OutputMatrixType = NS::RowMajMatrix<OutputMatrixDataType>;
    using OutputType = std::tuple<GrainT, OutputMatrixType>;

    //Get the transformer
    const auto* state_tensor(ctx->Input<Tensor>(0));
    const uint8_t* const state_data(state_tensor->Data<uint8_t>());
    Microsoft::Featurizer::Archive archive(state_data, state_tensor->Shape().GetDims()[0]);
    typename EstimatorT::TransformerType transformer(archive);

    // Get the Grains
    const auto* grains_tensor(ctx->Input<Tensor>(1));
    const std::string* grains_data(grains_tensor->Data<std::string>());
    const int64_t grains_num = grains_tensor->Shape()[1];

    // Get the Target
    const auto* target_tensor(ctx->Input<Tensor>(2));
    const T* target_data(target_tensor->Data<T>());

    // Prepare the OutputGrains
    TensorShape output_grains_shape = grains_tensor->Shape();
    Tensor* output_grains_tensor(ctx->Output(0, output_grains_shape));
    std::string* output_grains_data = output_grains_tensor->MutableData<std::string>();

    // Prepare the Output
    const int64_t output_dim_0 = grains_tensor->Shape()[0];

    double* output_data;
    bool has_allocate_output_data = false;
    std::function<void(OutputType)> callback_fn;
    callback_fn = [&ctx, &output_grains_data, &output_data, &has_allocate_output_data, &output_dim_0](OutputType value) -> void {
      GrainT output_grains = std::get<GrainT>(value);
      OutputMatrixType output_matrix = std::get<OutputMatrixType>(value);
      if (!has_allocate_output_data) {
        TensorShape output_shape({output_dim_0, output_matrix.rows(), output_matrix.cols()});
        Tensor* output_tensor(ctx->Output(1, output_shape));
        output_data = output_tensor->MutableData<double>();
        has_allocate_output_data = true;
      }
      output_grains_data = std::transform(
                             output_grains.begin(),
                             output_grains.end(),
                             output_grains_data,
                             [](std::string const & s) -> std::string {
                               return s;
                             }
                           );
      output_data = std::transform(
                      output_matrix.data(),
                      output_matrix.data() + output_matrix.size(),
                      output_data,
                      [](nonstd::optional<T> const & x) -> double {
                        return x.has_value() ? static_cast<double>(*x) : NS::Traits<double>::CreateNullValue();
                      }
                    );
    };

    // Transform
    std::vector<std::string> grains;
    grains.reserve(grains_num);
    for (int64_t i = 0; i < output_dim_0; ++i) {
      //Prepare Input and Output
      grains.clear();
      std::copy(grains_data, grains_data + grains_num, std::back_inserter(grains));
      GrainedInputType const input_tuple = std::make_tuple(grains, *target_data);
      //Execute
      transformer.execute(input_tuple, callback_fn);
      //Pointer Increment
      target_data++;
      grains_data += grains_num;
      if (i == 2)
        transformer.flush(callback_fn);
    }
    transformer.flush(callback_fn);
  }
};

class LagLeadOperatorTransformer final : public OpKernel {
 public:
  explicit LagLeadOperatorTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    utils::MLTypeCallDispatcher<LagLeadOperatorTransformerImpl, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t,
                                int64_t, uint64_t, float, double>
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
        .TypeConstraint("T", {DataTypeImpl::GetTensorType<int8_t>(),
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
