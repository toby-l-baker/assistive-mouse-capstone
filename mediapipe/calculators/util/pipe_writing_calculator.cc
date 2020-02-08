#include <sys/types.h>
#include <fcntl.h>
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
}
class PipeWritingCalculator : public CalculatorBase {
    public:
        PipeWritingCalculator() {}
        ~PipeWritingCalculator() override {}
        static ::mediapipe::Status GetContract(CalculatorContract* cc);
        ::mediapipe::Status Open(CalculatorContext* cc) override;
        ::mediapipe::Status Process(CalculatorContext* cc) override;
    private:
        int fd;

};
REGISTER_CALCULATOR(PipeWritingCalculator);

::mediapipe::Status PipeWritingCalculator::GetContract(CalculatorContract* cc) {
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
    if (cc->Inputs().HasTag(kGestureTag)) {
        cc->Inputs().Tag(kGestureTag).Set<int>();
        //cc->Inputs().Tag(kNormLandmarksTag).SetAny();
    }
    return ::mediapipe::OkStatus();
}

::mediapipe::Status PipeWritingCalculator::Open(CalculatorContext* cc) {
  cc->SetOffset(TimestampDiff(0));
  fd = open("/Users/Kururuken/desktop/MediaPipe_Gesture/test", O_WRONLY | O_CREAT);
  return ::mediapipe::OkStatus();
}

::mediapipe::Status PipeWritingCalculator::Process(CalculatorContext* cc) {
    const auto& landmarks = cc->Inputs().Tag(kLandmarksTag).Get<std::vector<NormalizedLandmark>>();
    const auto& gesture = cc->Inputs().Tag(kGestureTag).Get<int>();
    std::string res = "";
    res += std::to_string(landmarks[0].x());
    res += ",";
    res += std::to_string(landmarks[0].y());
    res += ",";
    res += std::to_string(gesture);
    res += "\n";
    write(fd, res.c_str(), res.length());
    return ::mediapipe::OkStatus();
}
}
