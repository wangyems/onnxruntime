// ----------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
// ----------------------------------------------------------------------

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "Featurizers/../Archive.h"
#include "Featurizers/LagLeadOperatorFeaturizer.h"
#include "Featurizers/TestHelpers.h"

namespace NS = Microsoft::Featurizer;

namespace onnxruntime {
namespace test {
namespace {

using InputTupleType = std::tuple<std::vector<std::string> const &, int64_t const &>;
using EstimatorT = NS::Featurizers::GrainedLagLeadOperatorEstimator<int64_t>;

std::vector<uint8_t> GetStream(EstimatorT& estimator, const std::vector<InputTupleType>& trainingBatches) {
  NS::TestHelpers::Train<EstimatorT, InputTupleType>(estimator, trainingBatches);
  auto pTransformer = estimator.create_transformer();
  NS::Archive ar;
  pTransformer->save(ar);
  return ar.commit();
}

} // namespace

TEST(FeaturizersTests, Grained_LagLead_2_grain_horizon_2_lead_1_lead_2) {
  //parameter setting
  using InputType = int64_t;
  using GrainType = std::vector<std::string>;
  using OutputMatrixDataType = NS::Traits<InputType>::nullable_type;
  using TransformedType = std::tuple<std::vector<std::string>, NS::RowMajMatrix<OutputMatrixDataType>>;
  NS::AnnotationMapsPtr                                            pAllColumnAnnotations(NS::CreateTestAnnotationMapsPtr(1));
  NS::Featurizers::GrainedLagLeadOperatorEstimator<InputType>      estimator(pAllColumnAnnotations, 2, {1, 2});
  using GrainedInputType = std::tuple<GrainType, InputType>;
  const GrainType grain1({"one"});
  const GrainedInputType tup1 = std::make_tuple(grain1, static_cast<InputType>(10));
  const GrainedInputType tup2 = std::make_tuple(grain1, static_cast<InputType>(11));
  const GrainedInputType tup3 = std::make_tuple(grain1, static_cast<InputType>(12));
  const GrainType grain2({"two"});
  const GrainedInputType tup4 = std::make_tuple(grain2, static_cast<InputType>(20));
  const GrainedInputType tup5 = std::make_tuple(grain2, static_cast<InputType>(21));
  const GrainedInputType tup6 = std::make_tuple(grain2, static_cast<InputType>(22));
  auto training_batch = NS::TestHelpers::make_vector<std::tuple<GrainType const &, InputType const &>>(tup1, tup2, tup3, tup4, tup5, tup6);

  auto stream = GetStream(estimator, training_batch);
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("LagLeadOperatorTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Grains", {6, 1}, {"one", "one", "one", "two", "two", "two"});
  test.AddInput<int64_t>("Target", {6}, {10, 11, 12, 20, 21, 22});
  test.AddOutput<std::string>("OutputGrains", {6, 1}, {"one", "one", "one", "two", "two", "two"});
  test.AddOutput<double>("Output", {6, 2, 2}, {10, 11, 11, 12,
                                               11, 12, 12, NS::Traits<double>::CreateNullValue(),
                                               12, NS::Traits<double>::CreateNullValue(), NS::Traits<double>::CreateNullValue(), NS::Traits<double>::CreateNullValue(),
                                               20, 21, 21, 22,
                                               21, 22, 22, NS::Traits<double>::CreateNullValue(),
                                               22, NS::Traits<double>::CreateNullValue(), NS::Traits<double>::CreateNullValue(), NS::Traits<double>::CreateNullValue()});

  test.Run();
}

//namespace {

// using InputType = double;
// using TransformerT = NS::Featurizers::LagLeadOperatorTransformer<InputType>;

// std::vector<uint8_t> GetTransformerStream(TransformerT& transformer) {
//   NS::Archive ar;
//   transformer.save(ar);
//   return ar.commit();
// }

// } // namespace

// TEST(FeaturizersTests, LagLeadOperator_Transformer_Draft) {
//   //parameter setting
//   TransformerT transformer(1, {-1, 1});

//   auto stream = GetTransformerStream(transformer);
//   auto dim = static_cast<int64_t>(stream.size());
//   OpTester test("LagLeadOperatorTransformer", 1, onnxruntime::kMSFeaturizersDomain);
//   test.AddInput<uint8_t>("State", {dim}, stream);
//   test.AddInput<std::string>("Grains", {5, 1}, {"a", "a", "a", "a", "a"});
//   test.AddInput<double>("Target", {5}, {10, 11, 12, 13, 14});
//   test.AddOutput<double>("Output", {5, 2, 1}, {NS::Traits<double>::CreateNullValue(), 11, 10, 12, 11, 13, 12, 14, 13, NS::Traits<double>::CreateNullValue()});

//   test.Run();
// }

}  // namespace test
}  // namespace onnxruntime
