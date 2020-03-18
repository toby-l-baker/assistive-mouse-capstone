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

// point1: (x1, y1)  point2(x2, y2)
double distance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}
// line1: starts from (x1, y1), ends at (x2, y2)
// line2: starts from (x3, y3), ends at (x4, y4)
double angle_cos(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4) {
    double vector1x = x2 - x1;
    double vector1y = y2 - y1;
    double vector2x = x4 - x3;
    double vector2y = y4 - y3;
    double vector1_length = distance(x1, y1, x2, y2);
    double vector2_length = distance(x3, y3, x4, y4);
    return (vector1x * vector2x + vector1y * vector2y) / (vector1_length * vector2_length);
}

}  // namespace

class GestureDetectionCalculator : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Open(CalculatorContext* cc) override;

  ::mediapipe::Status Process(CalculatorContext* cc) override;

};
REGISTER_CALCULATOR(GestureDetectionCalculator);

::mediapipe::Status GestureDetectionCalculator::GetContract(
    CalculatorContract* cc) {
  RET_CHECK(cc->Inputs().HasTag(kNormalizedLandmarksTag));
  RET_CHECK(cc->Outputs().HasTag(kGestureTag));
  // TODO: Also support converting Landmark to Detection.
  cc->Inputs().Tag(kNormalizedLandmarksTag).Set<NormalizedLandmarkList>();
  cc->Outputs().Tag(kGestureTag).Set<int>();

  return ::mediapipe::OkStatus();
}

::mediapipe::Status GestureDetectionCalculator::Open(
    CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));

  return ::mediapipe::OkStatus();
}

::mediapipe::Status GestureDetectionCalculator::Process(
    CalculatorContext* cc) {
  const auto& landmarks =
      cc->Inputs().Tag(kNormalizedLandmarksTag).Get<NormalizedLandmarkList>();
  RET_CHECK_GT(landmarks.landmark_size(), 0)
      << "Input landmark vector is empty.";

  // 0: wrist
  // 4: thumb
  // 8: index finger
  // 12: middle finger
  // 16: ring finger
  // 20: little finger
  double indexTipWristDistance = distance(landmarks.landmark(0).x(), landmarks.landmark(0).y(), landmarks.landmark(8).x(), landmarks.landmark(8).y());
  double indexBottomWristDistance = distance(landmarks.landmark(0).x(), landmarks.landmark(0).y(), landmarks.landmark(5).x(), landmarks.landmark(5).y());

  double middleTipWristDistance = distance(landmarks.landmark(0).x(), landmarks.landmark(0).y(), landmarks.landmark(12).x(), landmarks.landmark(12).y());
  double middleBottomWristDistance = distance(landmarks.landmark(0).x(), landmarks.landmark(0).y(), landmarks.landmark(9).x(), landmarks.landmark(9).y());

  double ringTipWristDistance = distance(landmarks.landmark(0).x(), landmarks.landmark(0).y(), landmarks.landmark(16).x(), landmarks.landmark(16).y());
  double ringBottomWristDistance = distance(landmarks.landmark(0).x(), landmarks.landmark(0).y(), landmarks.landmark(13).x(), landmarks.landmark(13).y());

  double littleTipWristDistance = distance(landmarks.landmark(0).x(), landmarks.landmark(0).y(), landmarks.landmark(20).x(), landmarks.landmark(20).y());
  double littleBottomWristDistance = distance(landmarks.landmark(0).x(), landmarks.landmark(0).y(), landmarks.landmark(17).x(), landmarks.landmark(17).y());

  double thumbAngleCos = angle_cos(landmarks.landmark(2).x(), landmarks.landmark(2).y(), landmarks.landmark(5).x(), landmarks.landmark(5).y(),
          landmarks.landmark(2).x(), landmarks.landmark(2).y(), landmarks.landmark(4).x(), landmarks.landmark(4).y());
  double thumbTipWristDistance = distance(landmarks.landmark(0).x(), landmarks.landmark(0).y(), landmarks.landmark(4).x(), landmarks.landmark(4).y());
  double thumbBottomWristDistance = distance(landmarks.landmark(0).x(), landmarks.landmark(0).y(), landmarks.landmark(2).x(), landmarks.landmark(2).y());

  int fingers = 0;

  if (indexTipWristDistance / indexBottomWristDistance > 1.5) {
      fingers++;
  }
  if (middleTipWristDistance / middleBottomWristDistance > 1.5) {
      fingers++;
  }
  if (ringTipWristDistance / ringBottomWristDistance > 1.5) {
      fingers++;
  }
  if (littleTipWristDistance / littleBottomWristDistance > 1.5) {
      fingers++;
  }
  if (thumbTipWristDistance / thumbBottomWristDistance > 1.5) {
      fingers++;
  }


  if (thumbAngleCos < 0.3) {
      fingers *= -1;
  }

  int* i = new int;
  *i = fingers;
  cc->Outputs().Tag(kGestureTag).Add(i, cc->InputTimestamp());
  return ::mediapipe::OkStatus();
}

}  // namespace mediapipe
