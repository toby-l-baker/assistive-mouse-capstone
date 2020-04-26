// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// #include "absl/memory/memory.h"
// #include "absl/strings/str_cat.h"
// #include "absl/strings/str_join.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"
namespace mediapipe {

namespace {

constexpr char kLandmarkTag[] = "LANDMARK";
constexpr char kRenderDataTag[] = "RENDER_DATA";

}  // namespace

class CentroidToRenderDataCalculator : public CalculatorBase {
 public:
  CentroidToRenderDataCalculator() {}
  ~CentroidToRenderDataCalculator() override {}

  static ::mediapipe::Status GetContract(CalculatorContract* cc);

  ::mediapipe::Status Open(CalculatorContext* cc) override;

  ::mediapipe::Status Process(CalculatorContext* cc) override;

};
REGISTER_CALCULATOR(CentroidToRenderDataCalculator);

::mediapipe::Status CentroidToRenderDataCalculator::GetContract(
    CalculatorContract* cc) {
    cc->Inputs().Tag(kLandmarkTag).Set<Landmark>();
    RET_CHECK(cc->Outputs().HasTag(kRenderDataTag));
    cc->Outputs().Tag(kRenderDataTag).Set<RenderData>();
    return ::mediapipe::OkStatus();
}

::mediapipe::Status CentroidToRenderDataCalculator::Open(
    CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  return ::mediapipe::OkStatus();
}

::mediapipe::Status CentroidToRenderDataCalculator::Process(
    CalculatorContext* cc) {
    const auto& centroid = cc->Inputs().Tag(kLandmarkTag).Get<Landmark>();
    auto render_data = absl::make_unique<RenderData>();
    auto* render_data_annotation = render_data.get()->add_render_annotations();
    render_data_annotation->set_thickness(4.0);
    auto* landmark_data = render_data_annotation->mutable_point();

    /* LANDMARK */
    landmark_data->set_normalized(false);
    landmark_data->set_x(centroid.x());
    landmark_data->set_y(centroid.y());

    /* SEND IT */
    cc->Outputs().Tag(kRenderDataTag).Add(render_data.release(), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
}


}  // namespace mediapipe
