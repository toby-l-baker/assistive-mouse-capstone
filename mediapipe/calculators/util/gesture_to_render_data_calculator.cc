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
constexpr char kGestureTag[] = "GESTURE";
constexpr char kRenderDataTag[] = "RENDER_DATA";
}

class GestureToRenderDataCalculator : public CalculatorBase {
    public:
    GestureToRenderDataCalculator() {}
    ~GestureToRenderDataCalculator() override {}
    static ::mediapipe::Status GetContract(CalculatorContract* cc);
    ::mediapipe::Status Open(CalculatorContext* cc) override;
    ::mediapipe::Status Process(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(GestureToRenderDataCalculator);

::mediapipe::Status GestureToRenderDataCalculator::GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag(kGestureTag).Set<int>();
    cc->Outputs().Tag(kRenderDataTag).Set<RenderData>();
    return ::mediapipe::OkStatus();
}

::mediapipe::Status GestureToRenderDataCalculator::Open(CalculatorContext* cc) {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
}

::mediapipe::Status GestureToRenderDataCalculator::Process(CalculatorContext* cc) {
    const auto& gesture = cc->Inputs().Tag(kGestureTag).Get<int>();
    auto render_data = absl::make_unique<RenderData>();
    auto* render_data_annotation = render_data.get()->add_render_annotations();
    auto* data = render_data_annotation->mutable_text();
    data->set_normalized(true);
    data->set_display_text(std::to_string(gesture));
    data->set_left(0.5);
    data->set_baseline(0.9);
    data->set_font_height(0.05);
    render_data_annotation->set_thickness(2.0);
    render_data_annotation->mutable_color()->set_r(255);

    cc->Outputs().Tag(kRenderDataTag).Add(render_data.release(), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
}
}
