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

using InputType = double;
using TransformerT = NS::Featurizers::LagLeadOperatorTransformer<InputType>;

std::vector<uint8_t> GetTransformerStream(TransformerT& transformer) {
  NS::Archive ar;
  transformer.save(ar);
  return ar.commit();
}

} // namespace

TEST(FeaturizersTests, LagLeadOperator_Transformer_Draft) {
  //parameter setting
  TransformerT transformer(1, {-1, 1});

  auto stream = GetTransformerStream(transformer);
  auto dim = static_cast<int64_t>(stream.size());
  OpTester test("LagLeadOperatorTransformer", 1, onnxruntime::kMSFeaturizersDomain);
  test.AddInput<uint8_t>("State", {dim}, stream);
  test.AddInput<std::string>("Grains", {5, 1}, {"a", "a", "a", "a", "a"});
  test.AddInput<double>("Target", {5}, {10, 11, 12, 13, 14});
  test.AddOutput<double>("Output", {5, 2, 1}, {NS::Traits<double>::CreateNullValue(), 11, 10, 12, 11, 13, 12, 14, 13, NS::Traits<double>::CreateNullValue()});

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
