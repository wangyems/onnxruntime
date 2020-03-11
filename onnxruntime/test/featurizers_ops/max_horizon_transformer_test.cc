// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(FeaturizersTests, MaxHorizonTransformer) {
  OpTester test("MaxHorizonTransformer", 1, onnxruntime::kMSFeaturizersDomain);

  std::vector<int64_t> times_input = {1, 2, 3};
  std::vector<std::string> keys_input = {"a1", "b1", "c1", "a2", "b2", "c2"};
  std::vector<std::string> data_input = {"d1", "d2", "d3"};
  std::vector<std::uint32_t> max_horizon = {2};

  ASSERT_TRUE(keys_input.size() % times_input.size() == 0);
  ASSERT_TRUE(data_input.size() % times_input.size() == 0);
  test.AddInput<int64_t>("Times", {static_cast<int64_t>(times_input.size())}, times_input);
  test.AddInput<std::string>("Keys", {static_cast<int64_t>(times_input.size()), static_cast<int64_t>(keys_input.size() / times_input.size())}, keys_input);
  test.AddInput<std::string>("Data", {static_cast<int64_t>(times_input.size()), static_cast<int64_t>(data_input.size() / times_input.size())}, data_input);
  test.AddInput<uint32_t>("MaxHorizon", {static_cast<int64_t>(1)}, max_horizon);

  std::vector<int64_t> times_output = {1, 1, 2, 2, 3, 3};
  std::vector<std::string> keys_output = {"a1", "a1", "b1", "b1", "c1", "c1", "a2", "a2", "b2", "b2", "c2", "c2"};
  std::vector<std::string> data_output = {"d1", "d1", "d2", "d2", "d3", "d3"};
  std::vector<uint32_t> horizon_origin = {1, 2, 1, 2, 1, 2};

  test.AddOutput<int64_t>("Times", {static_cast<int64_t>(times_output.size())}, times_output);
  test.AddOutput<std::string>("Keys", {static_cast<int64_t>(times_output.size()), static_cast<int64_t>(keys_output.size() / times_output.size())}, keys_output);
  test.AddOutput<std::string>("Data", {static_cast<int64_t>(times_output.size()), static_cast<int64_t>(data_output.size() / times_output.size())}, data_output);
  test.AddOutput<uint32_t>("HorizonOrigin", {static_cast<int64_t>(times_output.size())}, horizon_origin);

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
