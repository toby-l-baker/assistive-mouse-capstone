#include <chrono>
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
#define PI 3.141592653589793238462643383279502884L
namespace mediapipe {
namespace {
constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kNormLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kGestureTag[] = "GESTURE";

// point1: (x1, y1)  point2(x2, y2)
double distance(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}
// vector1: starts from (x1, y1), ends at (x2, y2)
// vector2: starts from (x3, y3), ends at (x4, y4)
double angle_cos(double x1, double y1, double x2, double y2, double x3, double y3, double x4, double y4) {
    double vector1x = x2 - x1;
    double vector1y = y2 - y1;
    double vector2x = x4 - x3;
    double vector2y = y4 - y3;
    double vector1_length = distance(x1, y1, x2, y2);
    double vector2_length = distance(x3, y3, x4, y4);
    return (vector1x * vector2x + vector1y * vector2y) / (vector1_length * vector2_length);
}
// vector1: starts from (x1, y1), ends at (x2, y2)
// vector2: starts from (0, 0), ends at (0, 1)
double angle_cos(double x1, double y1, double x2, double y2) {
    double vector1x = x2 - x1;
    double vector1y = y2 - y1;
    double vector2x = 0;
    double vector2y = 1;
    double vector1_length = distance(x1, y1, x2, y2);
    double vector2_length = 1.0;
    return (vector1x * vector2x + vector1y * vector2y) / (vector1_length * vector2_length);
}

// vector1: starts from (x1, y1), ends at (x2, y2)
// vector2: starts from (0, 0), ends at (0, 1)
double angle_sin(double x1, double y1, double x2, double y2) {
    double cosine = angle_cos(x1, y1, x2, y2);
    double vector1x = x2 - x1;
    if (vector1x <= 0) {
        return -1.0 * sqrt(1 - pow(cosine, 2));
    } else {
        return sqrt(1 - pow(cosine, 2));
    }
}

// vector1: starts from (x1, y1), ends at (x2, y2)
// vector2: starts from (0, 0), ends at (1, 0)
int angle(double x1, double y1, double x2, double y2) {
    double vector1x = x2 - x1;
    double vector1y = y2 - y1;
    double vector2x = 1;
    double vector2y = 0;
    double dot = vector1x * vector2x + vector1y * vector2y;
    double det = vector1x * vector2y - vector1y * vector2x;
    int angle = atan2(det, dot) / PI * 180;
    return (360 + angle) % 360;
}

/*
 * 0 1
 * 2 3
 */
double* get_matrix(double x1, double y1, double x2, double y2) {
    double* matrix = new double[4];
    double sine = angle_sin(x1, y1, x2, y2);
    double cosine = angle_cos(x1, y1, x2, y2);
    double len = distance(x1, y1, x2, y2);
    matrix[0] = cosine / len;
    matrix[1] = sine / len;
    matrix[2] = -1.0 * sine / len;
    matrix[3] = cosine / len;
    return matrix;
}

class MyLandmark {
    public:
    double _x, _y;
    MyLandmark(double x, double y) {
        _x = x;
        _y = y;
    }
    double x() {
        return _x;
    }
    double y() {
        return _y;
    }
};

std::vector<MyLandmark>* normalize(const std::vector<NormalizedLandmark>& landmarks) {
    // 0, 9
    std::vector<MyLandmark>* my_landmarks = new std::vector<MyLandmark>();
    double* matrix = get_matrix(landmarks[9].x(), landmarks[9].y(), landmarks[0].x(), landmarks[0].y());
    for (const auto& landmark : landmarks) {
        double x = landmark.x() - landmarks[0].x();
        double y = landmark.y() - landmarks[0].y();
        double x_new = x * matrix[0] + y * matrix[2];
        double y_new = x * matrix[1] + y * matrix[3];
        my_landmarks->push_back(*(new MyLandmark(x_new, y_new)));
    }
    delete[] matrix;
    return my_landmarks;
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
    if (cc->Inputs().HasTag(kNormLandmarksTag) || cc->Inputs().HasTag(kLandmarksTag)) {
        const auto& landmarks = cc->Inputs().Tag(kLandmarksTag).Get<std::vector<NormalizedLandmark>>();

        std::vector<MyLandmark>* normalized_landmarks_ptr = normalize(landmarks);
        std::vector<MyLandmark>& normalized_landmarks = *normalized_landmarks_ptr;

        // 0: wrist
        // 4: thumb
        // 8: index finger
        // 12: middle finger
        // 16: ring finger
        // 20: little finger


        int from5to4 = angle(normalized_landmarks[5].x(), normalized_landmarks[5].y(), normalized_landmarks[4].x(), normalized_landmarks[4].y());
        int from6to8 = angle(normalized_landmarks[6].x(), normalized_landmarks[6].y(), normalized_landmarks[8].x(), normalized_landmarks[8].y());
        int from10to12 = angle(normalized_landmarks[10].x(), normalized_landmarks[10].y(), normalized_landmarks[12].x(), normalized_landmarks[12].y());
        int from14to16 = angle(normalized_landmarks[14].x(), normalized_landmarks[14].y(), normalized_landmarks[16].x(), normalized_landmarks[16].y());
        int from18to20 = angle(normalized_landmarks[18].x(), normalized_landmarks[18].y(), normalized_landmarks[20].x(), normalized_landmarks[20].y());
        
        int gesture = 0;
        // move: gesture = 1
        if (abs(from6to8 - 90) + abs(from10to12 - 90) + abs(from14to16 - 90) + abs(from18to20 - 90) <= 80) {
            gesture = 1;
        }
        // scroll down: gesture = 2
        else if (abs(from6to8 - 90) + abs(from10to12 - 90) <= 40 && from14to16 > 150 && from18to20 > 150) {
            gesture = 2;
        }
        // scroll up: gesture = 3
        else if (abs(from6to8 - 90) <= 20 && from10to12 > 150 && from14to16 > 150 && from18to20 > 150) {
            gesture = 3;
        }
        // left click: gesture = 4
        else if (from6to8 > 150 && from10to12 > 150 && from14to16 > 150 && from18to20 > 150) {
            gesture = 4;
        }
        // right click: gesture = 5
        else if (from5to4 > 190) {
            gesture = 5;
        }

        if ((from6to8 - 120) + (from10to12 - 120) + (from14to16 - 120) + (from18to20 - 120) >= 10) {
            gesture += 10;
        }
        //std::cout << from5to4 << " " << from6to8 << " " << from10to12 << " " << from14to16 << " " << from18to20 << " " <<  gesture << "\n";

        delete normalized_landmarks_ptr;

        int* i = new int;
        *i = gesture;
        cc->Outputs().Tag(kGestureTag).Add(i, cc->InputTimestamp());
    }
    return ::mediapipe::OkStatus();
}
}
