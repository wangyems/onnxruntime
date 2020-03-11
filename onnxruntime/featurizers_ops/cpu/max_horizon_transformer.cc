// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/framework/op_kernel.h"

#include <cstdlib>
#include <limits>

namespace onnxruntime {
namespace featurizers {

template <typename T>
void MaxHorizonTransformerImpl(OpKernelContext* ctx) {
  const auto& input_times = *ctx->Input<Tensor>(0);
  const auto& input_keys = *ctx->Input<Tensor>(1);
  const auto& input_data = *ctx->Input<Tensor>(2);
  const auto& input_max_horizon = *ctx->Input<Tensor>(3);

  const int64_t input_rows_num = input_times.Shape()[0];
  const int64_t keys_per_row = input_keys.Shape()[1];
  const int64_t data_num_per_row = input_data.Shape()[1];

  const int64_t* times_data = input_times.template Data<int64_t>();
  const T* const keys_data = input_keys.template Data<T>();
  const T* const data_data = input_data.template Data<T>();
  const uint32_t* max_horizon_data = input_max_horizon.template Data<uint32_t>();

  int64_t output_rows_num = (*max_horizon_data) * input_rows_num;
  TensorShape rows_shape({output_rows_num});
  TensorShape keys_shape({output_rows_num, keys_per_row});
  TensorShape data_shape({output_rows_num, data_num_per_row});

  auto* times_output = ctx->Output(0, rows_shape)->template MutableData<int64_t>();
  auto* keys_output = ctx->Output(1, keys_shape)->template MutableData<T>();
  auto* data_output = ctx->Output(2, data_shape)->template MutableData<T>();
  auto* horizon_origin_output = ctx->Output(3, rows_shape)->template MutableData<uint32_t>();

  //todo: may need locality optimization
  for (int64_t row_idx = 0; row_idx < input_rows_num; ++row_idx) {
    for (uint32_t horizon_idx = 0; horizon_idx < *max_horizon_data; ++horizon_idx) {
      *times_output++ = *times_data;
      for (int64_t keys_col_idx = 0; keys_col_idx < keys_per_row; ++ keys_col_idx)
        keys_output[keys_col_idx * output_rows_num + row_idx * (*max_horizon_data) + horizon_idx] = keys_data[keys_col_idx * input_rows_num + row_idx];
      for (int64_t data_num_col_idx = 0; data_num_col_idx < data_num_per_row; ++data_num_col_idx)
        data_output[data_num_col_idx * output_rows_num + row_idx * (*max_horizon_data) + horizon_idx] = data_data[data_num_col_idx * input_rows_num + row_idx];
      *horizon_origin_output++ = horizon_idx + 1;
    }
    times_data++;
  }
}

class MaxHorizonTransformer final : public OpKernel {
 public:
  explicit MaxHorizonTransformer(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* ctx) const override {
    MaxHorizonTransformerImpl<std::string>(ctx);
    return Status::OK();
  }
};

ONNX_OPERATOR_KERNEL_EX(
    MaxHorizonTransformer,
    kMSFeaturizersDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T0", DataTypeImpl::GetTensorType<int64_t>())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<std::string>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint32_t>()),
    MaxHorizonTransformer
  );
}  // namespace featurizers
}  // namespace onnxruntime
