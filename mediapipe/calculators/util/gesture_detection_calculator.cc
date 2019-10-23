#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"
namespace mediapipe {
namespace {
constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";
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
}
class GestureDetectionCalculator : public CalculatorBase {
    public:
        GestureDetectionCalculator() {}
        ~GestureDetectionCalculator() override {}
        static ::mediapipe::Status GetContract(CalculatorContract* cc);
        ::mediapipe::Status Open(CalculatorContext* cc) override;
        ::mediapipe::Status Process(CalculatorContext* cc) override;

};
REGISTER_CALCULATOR(GestureDetectionCalculator);

::mediapipe::Status GestureDetectionCalculator::GetContract(CalculatorContract* cc) {
    RET_CHECK(cc->Inputs().HasTag(kLandmarksTag) ||
            cc->Inputs().HasTag(kNormLandmarksTag))
        << "None of the input streams are provided.";
    RET_CHECK(!(cc->Inputs().HasTag(kLandmarksTag) &&
                cc->Inputs().HasTag(kNormLandmarksTag)))
        << "Can only one type of landmark can be taken. Either absolute or "
        "normalized landmarks.";

    if (cc->Inputs().HasTag(kLandmarksTag)) {
        //cc->Inputs().Tag(kLandmarksTag).Set<std::vector<Landmark>>();
        cc->Inputs().Tag(kLandmarksTag).SetAny();
    }
    if (cc->Inputs().HasTag(kNormLandmarksTag)) {
        //cc->Inputs().Tag(kNormLandmarksTag).Set<std::vector<NormalizedLandmark>>();
        cc->Inputs().Tag(kNormLandmarksTag).SetAny();
    }
    cc->Outputs().Tag(kGestureTag).Set<int>();
    return ::mediapipe::OkStatus();
}

::mediapipe::Status GestureDetectionCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  return ::mediapipe::OkStatus();
}

::mediapipe::Status GestureDetectionCalculator::Process(CalculatorContext* cc) {
    /*
    if (cc->Inputs().HasTag(kLandmarksTag)) {
        const auto& landmarks = cc->Inputs().Tag(kLandmarksTag).Get<std::vector<Landmark>>();
        std::cout << landmarks[0].x() << std::endl;
        int i = 1;
        cc->Outputs().Tag(kGestureTag).Add(&i, cc->InputTimestamp());
    }
    */
    if (cc->Inputs().HasTag(kNormLandmarksTag) || cc->Inputs().HasTag(kLandmarksTag)) {
        const auto& landmarks = cc->Inputs().Tag(kLandmarksTag).Get<std::vector<NormalizedLandmark>>();
        // 0: wrist
        // 4: thumb
        // 8: index finger
        // 12: middle finger
        // 16: ring finger
        // 20: little finger
        double indexTipWristDistance = distance(landmarks[0].x(), landmarks[0].y(), landmarks[8].x(), landmarks[8].y());
        double indexBottomWristDistance = distance(landmarks[0].x(), landmarks[0].y(), landmarks[5].x(), landmarks[5].y());

        double middleTipWristDistance = distance(landmarks[0].x(), landmarks[0].y(), landmarks[12].x(), landmarks[12].y());
        double middleBottomWristDistance = distance(landmarks[0].x(), landmarks[0].y(), landmarks[9].x(), landmarks[9].y());

        double ringTipWristDistance = distance(landmarks[0].x(), landmarks[0].y(), landmarks[16].x(), landmarks[16].y());
        double ringBottomWristDistance = distance(landmarks[0].x(), landmarks[0].y(), landmarks[13].x(), landmarks[13].y());

        double littleTipWristDistance = distance(landmarks[0].x(), landmarks[0].y(), landmarks[20].x(), landmarks[20].y());
        double littleBottomWristDistance = distance(landmarks[0].x(), landmarks[0].y(), landmarks[17].x(), landmarks[17].y());

        double thumbAngleCos = angle_cos(landmarks[0].x(), landmarks[0].y(), landmarks[17].x(), landmarks[17].y(),
                landmarks[0].x(), landmarks[0].y(), landmarks[4].x(), landmarks[4].y());

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
        if (thumbAngleCos < 0.5) {
            fingers++;
        }

        int* i = new int;
        *i = fingers;
        cc->Outputs().Tag(kGestureTag).Add(i, cc->InputTimestamp());
    }
    return ::mediapipe::OkStatus();
}
}
