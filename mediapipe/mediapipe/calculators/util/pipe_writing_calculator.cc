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

#include <memory>
#include <fcntl.h>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include <iostream>

namespace mediapipe {

namespace {

constexpr char kNormalizedLandmarksTag[] = "LANDMARKS";
constexpr char kGestureTag[] = "GESTURE";

}  // namespace

class PipeWritingCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Open(CalculatorContext* cc) override;

  ::mediapipe::Status Process(CalculatorContext* cc) override;

 private:
  int fd;
};
REGISTER_CALCULATOR(PipeWritingCalculator);

::mediapipe::Status PipeWritingCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kNormalizedLandmarksTag));
  RET_CHECK(cc->Inputs().HasTag(kGestureTag));
  // TODO: Also support converting Landmark to Detection.
  cc->Inputs().Tag(kNormalizedLandmarksTag).Set<NormalizedLandmarkList>();
  cc->Inputs().Tag(kGestureTag).Set<int>();

  return ::mediapipe::OkStatus();
}

::mediapipe::Status PipeWritingCalculator::Open(
    CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  fd = open("/home/weihao/Mouse", O_WRONLY);
  return ::mediapipe::OkStatus();
}

::mediapipe::Status PipeWritingCalculator::Process(
    CalculatorContext* cc) {
  const auto& landmarks =
      cc->Inputs().Tag(kNormalizedLandmarksTag).Get<NormalizedLandmarkList>();
  RET_CHECK_GT(landmarks.landmark_size(), 0)
      << "Input landmark vector is empty.";
  const auto& gesture = cc->Inputs().Tag(kGestureTag).Get<int>();
  std::string res = "";
  double x_avg = (landmarks.landmark(0).x() + landmarks.landmark(5).x() + landmarks.landmark(9).x() + landmarks.landmark(13).x() + landmarks.landmark(17).x()) / 5;
  double y_avg = (landmarks.landmark(0).y() + landmarks.landmark(5).y() + landmarks.landmark(9).y() + landmarks.landmark(13).y() + landmarks.landmark(17).y()) / 5;
  //double x_avg = (landmarks[8].x() + landmarks[12].x() + landmarks[16].x() + landmarks[20].x()) / 4;
  //double y_avg = (landmarks[8].y() + landmarks[12].y() + landmarks[16].y() + landmarks[20].y()) / 4;
  //res += std::to_string(landmarks[0].x());
  res += std::to_string(x_avg);
  res += ",";
  //res += std::to_string(landmarks[0].y());
  res += std::to_string(y_avg);
  res += ",";
  res += std::to_string(gesture);
  res += "\n";
  write(fd, res.c_str(), res.length());
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
