#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/udp.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/rect.pb.h"

using namespace cv;
using namespace std;

// Image Capture Parameters
#define BLUR_KERNEL_SIZE 5

// Calibration Parameters
#define BOX_SIZE 50

// Hand Histogram Parameters
#define H_BINS 12
#define S_BINS 15
#define HIST_SENSITIVITY 0.01

// Thresholding Parameters
#define HS_THRESH 50
#define EROSION_IT 2
#define DILATION_IT 4

// Tracker Parameters
#define EXPANSION_VAL 50
#define SAMPLE_BOX_SIZE 30

/*** GLOBALS ***/

// width and height of the frame
double dHeight;
double dWidth;

// Hand Histogram
cv::Mat hand_hist;

// Setup for various histogram values
int ch[] = {0, 1};
int hist_size[] = {H_BINS, S_BINS};
float h_ranges[] = {0, 180};
float s_ranges[] = {0, 255};
const float* ranges[] = { h_ranges, s_ranges };

// For drawing on frames
cv::Scalar blue (255, 0, 0);
cv::Scalar green (0, 255, 0);
cv::Scalar red (0, 0, 255);

/*** Helper Functions ***/

// Initialises our histogram using a hand sample when it first enters the frame
cv::Mat get_histogram(const cv::Mat& roi) {
     cv::Mat hand_histogram;
     cv::calcHist(&roi, 1, ch, cv::Mat(), //do not use a mask
                    hand_histogram, 2, hist_size, ranges, true); 
     cv::normalize(hand_histogram, hand_histogram, 0, 255, cv::NORM_MINMAX);
     return hand_histogram;
}

// Adapts the histogram based off of each new frame to account for changes in lighting
// takes in a sample and the current histogram and returns a new one
cv::Mat adapt_histogram(const cv::Mat& roi, cv::Mat& histogram) {
    cv::Mat new_hist;
    cv::calcHist(&roi, 1, ch, cv::Mat(), //do not use a mask
                    new_hist, 2, hist_size, ranges, true); 
    cv::normalize(new_hist, new_hist, 0, 255, cv::NORM_MINMAX);
    return histogram*(1-HIST_SENSITIVITY) + HIST_SENSITIVITY*new_hist;
}

// Clips an integer value to be in some range lower < n < upper
int clip(int n, int lower, int upper) {
  return std::max(lower, std::min(n, upper));
}

// Takes in an hsv frame, performs some morphology and thresholding
// it then finds all contours in the image matching the hand and returns
// them
vector<vector<Point>> get_contours(cv::Mat hsv_frame) {
    cv::Mat temp;
    // cv::Mat filtered;
    cv::Mat backproj;
    cv::calcBackProject(&hsv_frame, 1, ch, hand_hist, backproj, ranges); // see how well the pixels fit our histogram
    cv::Mat erosion_kernel = cv::getStructuringElement(cv::MORPH_RECT, Size(7, 7)); 
    cv::Mat dilation_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(8, 8));

    // Filtering, morphological operations and thresholding
    cv::filter2D(backproj, temp, -1, // dimension the same as source image
                getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(6,6))); //convolves the image with this kernel
    cv::threshold(temp, temp, HS_THRESH, 255, 0); // 0 is binary thresholding
    cv::erode(temp, temp, erosion_kernel, cv::Point(-1, -1), EROSION_IT); //anchor, iterations
    cv::dilate(temp, temp, dilation_kernel, cv::Point(-1, -1), DILATION_IT);
    cv::medianBlur(temp, temp, 5);

    // Get contours
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    cv::findContours(temp, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    // //show the thresholded frame
    // cv::imshow("Thresholded Output", temp);
    // //show the output of filter 2D
    // cv::imshow("Filtered Back Projection", filtered);
    return contours;
}

/*** Hand Object ***/

class Hand {
    public:
        cv::Point centroid; // centroid of the contour
        cv::Rect bound; // bounding box of the contour
        double area; // area of the contour
        // double timestamp;
        cv::Point velocity; // the relative movement of the centroid compared to the previous frame
        void set_state(cv::Point, cv::Rect, double); // sets the above variables
        void update_velocity(Hand); // sets the velocity using the previous state
} hand;

void Hand::set_state (cv::Point centroid_, cv::Rect bound_, double area_) {
    centroid = centroid_;
    bound = bound_;
    area = area_;
    // timestamp = timestamp_;
}

void Hand::update_velocity(Hand old) {
    // float dt = timestamp - old.timestamp;
    velocity = centroid - old.centroid;
}

/*** Hand Tracker Object ***/

class HandTracker {
    public:
        Hand old_hand;
        Hand new_hand;
        cv::Rect predict_position(void); // predicts a region where the hand will be based on the prior velocity
        cv::Rect update_position(cv::Mat, cv::Rect roi_mp, bool mp_available); // locates the hand in the bounded region defined by predict_position
        void initialise(cv::Mat frame);
} tracker;

cv::Rect HandTracker::predict_position() {
    float vx = old_hand.velocity.x;
    float vy = old_hand.velocity.y;
    cv::Rect prediction (old_hand.bound.x, old_hand.bound.y, old_hand.bound.width+EXPANSION_VAL, old_hand.bound.height+EXPANSION_VAL); // create a wider region to search for the hand in
    prediction.x = prediction.x + vx - EXPANSION_VAL/2; // shift the origin of the rect so their centers align 
    prediction.y = prediction.y + vy - EXPANSION_VAL/2; 
    prediction.x = clip(prediction.x, 0, dWidth); // co-ordinate shift to global frame and clamping
    prediction.y = clip(prediction.y, 0, dHeight);

    int max_width = dWidth - prediction.x;
    int max_height = dHeight - prediction.y;
    prediction.width = clip(prediction.width, 0, max_width);
    prediction.height = clip(prediction.height, 0, max_height);
    return prediction;
}

// Returns a hand struct of where the hand is in the frame currently
cv::Rect HandTracker::update_position(cv::Mat frame, cv::Rect roi_mp, bool mp_available) {
    // // To fix crashing
    // int width = clip(roi_mp.width, 0, dWidth-1);
    // int height = clip(roi_mp.height, 0, dHeight-1);
    cv::Rect roi (roi_mp.x, roi_mp.y, roi_mp.width, roi_mp.height);

    if (mp_available == false) {
        roi = predict_position();
    }
    vector<vector<Point>> contours;
    try {
        contours = get_contours(frame(roi)); // this will be in local coords
    } catch (const std::exception& e) {
        roi.x = 0;
        roi.y = 0;
        roi.width = dWidth-1;
        roi.height = dHeight-1;
        contours = get_contours(frame(roi)); // this will be in local coords
    }
    // Currently uses largest contour by area
    double max_area = 0.0;
    vector<Point> max_contour;
    for (int it = 0; it < contours.size(); it++) {
        double area = cv::contourArea(contours[it]);
        if (area > max_area) {
            max_area = area;
            max_contour = contours[it];
        }
    }

    // Draw a bounding box around the largest contour
    cv::Rect bound = cv::boundingRect(max_contour);
    bound.x = bound.x + roi.x; // co-ordinate shift to global frame and clamping
    bound.y = bound.y + roi.y;

    // Get the centroid
    cv::Moments m = cv::moments(max_contour);
    cv::Point centroid (m.m10/m.m00, m.m01/m.m00);
    centroid.x += roi.x;
    centroid.y += roi.y;

    // Set the state of the new hand
    new_hand.set_state(centroid, bound, max_area);

    return roi;
}

void HandTracker::initialise(cv::Mat frame) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV); // convert to hsv colour space
    vector<vector<Point>> conts = get_contours(hsv); // get all contours
    double max_area = 0.0;
    vector<Point> max_contour;
    // find max contour by area
    for (int it = 0; it < conts.size(); it++) {
        double area = cv::contourArea(conts[it]);
        if (area > max_area) {
            max_area = area;
            max_contour = conts[it];
        }
    }
    cv::Rect bound = cv::boundingRect(max_contour);
    // Get the centroid
    cv::Moments m = cv::moments(max_contour);
    cv::Point centroid (m.m10/m.m00, m.m01/m.m00);

    // Set the state of the new hand
    old_hand.set_state(centroid, bound, max_area);
    old_hand.velocity = {0, 0};
}


namespace mediapipe {

namespace {
constexpr char kInputFrameTag[] = "IMAGE";
constexpr char kLandmarksTag[] = "LANDMARKS";
constexpr char kLandmarkTag[] = "LANDMARK";
constexpr char kRectTag[] = "RECT";
constexpr char kNormRectTag[] = "NORM_RECT";
constexpr char kRectsTag[] = "RECTS";
constexpr char kHeightTag[] = "HEIGHT";
// constexpr char kImageTag[] = "IMAGE";
// constexpr char kImageGpuTag[] = "IMAGE_GPU";
constexpr char kWidthTag[] = "WIDTH";
}

const char *IP = "localhost";    // IP address of UDP server
const short PORT = 4000;         // port number of UDP server

class HandTrackingCalculator : public CalculatorBase {
private:
    udp::client *forwarder;
    bool image_frame_available;
    bool calibrated;
    cv::Mat frame;
    cv::Mat frame_blurred;
    cv::Mat hsv;
    cv::Mat hsv_roi; 
    HandTracker hand_tracker;
    cv::Rect sample_region; // for taking samples of skin colour
    cv::Rect bbox;
    cv::Rect prediction;
    double initial_area; // initial area of the hand - used to detect failures

    bool rect_available;
    bool landmarks_available;

    int hand_lost;

    // Setup for various histogram values
    // int ch[2];
    // int hist_size[2];
    // float h_ranges[2];
    // float s_ranges[2];
    // const float* ranges[2];

public:
    /* GetContract is the Input Handler */
    static ::mediapipe::Status GetContract(CalculatorContract* cc)
    {
        RET_CHECK(cc->Inputs().HasTag(kLandmarksTag)) << "No landmark input stream provided.";
        RET_CHECK(cc->Inputs().HasTag(kInputFrameTag)) << "No input image stream provided.";
        RET_CHECK((cc->Inputs().HasTag(kNormRectTag))) << "No norm rect input provided";

        /* If we have an image present then we set the image */
        if (cc->Inputs().HasTag(kInputFrameTag)) {
            // RET_CHECK(cc->Outputs().HasTag(kInputFrameTag));
            cc->Inputs().Tag(kInputFrameTag).Set<ImageFrame>();
            // cc->Outputs().Tag(kInputFrameTag).Set<ImageFrame>();
        }

        if(cc->Inputs().HasTag(kLandmarksTag))
        {
            cc->Inputs().Tag(kLandmarksTag).Set<NormalizedLandmarkList>();
        }

        if(cc->Inputs().HasTag(kNormRectTag))
        {
            cc->Inputs().Tag(kNormRectTag).Set<NormalizedRect>();
        }

        if (cc->Outputs().HasTag(kRectsTag)) {
            cc->Outputs().Tag(kRectsTag).Set<std::vector<Rect>>();
        }

        if (cc->Outputs().HasTag(kLandmarkTag)) {
            cc->Outputs().Tag(kLandmarkTag).Set<Landmark>();
        }

        return(::mediapipe::OkStatus());
    }

    ::mediapipe::Status Open(CalculatorContext* cc) override
    {
        cc->SetOffset(TimestampDiff(0));

        if(cc->Inputs().HasTag(kInputFrameTag))
        {
            image_frame_available = true;
        }
        else
        {
            image_frame_available = false;
        }

        forwarder = new udp::client(IP, PORT);

        rect_available = false;
        landmarks_available = false;
        calibrated = false; // uncalibrated histogram initially
        hand_lost = 0;

        return(::mediapipe::OkStatus());
    }

    ::mediapipe::Status Process(CalculatorContext* cc) override
    {
        /* If we have no image data just return */
        if (cc->Inputs().Tag(kInputFrameTag).IsEmpty()) {
            return ::mediapipe::OkStatus();
        }

        // Load input image
        const auto& input_img = cc->Inputs().Tag(kInputFrameTag).Get<ImageFrame>();
        frame = formats::MatView(&input_img); // frame used further down for CV things
        dWidth = frame.size().width;
        dHeight = frame.size().height;

        if (cc->Inputs().HasTag(kNormRectTag) && !cc->Inputs().Tag(kNormRectTag).IsEmpty()) {
            // if (hand_present_rect(cc, dWidth, dHeight)) {
            rect_available = true;
        } else {
            rect_available = false;
        }

        if (cc->Inputs().HasTag(kLandmarksTag) && !cc->Inputs().Tag(kLandmarksTag).IsEmpty()) {
            // if (hand_present_landmark(cc, dWidth, dHeight)) {
            landmarks_available = true;
        } else {
            landmarks_available = false;
        }


        string data;

        // if landmarks available get the region to sample
        if (landmarks_available) {
            sample_region = get_sample_rect(cc, dWidth, dHeight);
            if ((sample_region.width == 15) && (sample_region.height == 15)) {
                landmarks_available = false;
            }
        }
        
        // if MP has a  bounding box for the hand
        if (rect_available) {
            // Load normalized rect
            bbox = get_bounding_box(cc, dWidth, dHeight);
            if ((bbox.width == dWidth-1) && (bbox.height == dHeight-1)) {
                rect_available = false;
            }
        }

        /* CHECK IF THE HAND IS GONE */
        if ((!rect_available) && (!landmarks_available)) {
            hand_lost++;
        } else {
            hand_lost = 0;
        }

        if (hand_lost > 5) {
            calibrated = false;
        }
        /* For displaying rectangles */
        int num_rects = 1;
        if (landmarks_available) {
            num_rects = 2;
        }

        if (calibrated == false) {
            if (!landmarks_available) {
                VLOG(1) << "No Landmarks Available to Calibrate with" << cc->InputTimestamp();
                return ::mediapipe::OkStatus();
            }

            if (!rect_available) {
                VLOG(1) << "No Rect Available to Initialize Tracker with" << cc->InputTimestamp();
                return ::mediapipe::OkStatus();                
            }   

            cv::GaussianBlur(frame, frame_blurred, Size(BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0); // Blur image with 5x5 kernel
            cv::Mat roi = frame_blurred(sample_region); // grab the region of the frame we want
            cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV); // convert roi to HSV colour space
            hand_hist = get_histogram(hsv_roi); // get initial histogram
            calibrated = true; // hand histogram has been calibrated
            
            cv::Mat blurred_roi = frame_blurred(bbox); // search bounding region from MPipe
            hand_tracker.initialise(blurred_roi);
            initial_area = hand_tracker.old_hand.area;
        } else {
            cv::GaussianBlur(frame, frame_blurred, Size(BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0); // Blur image with 5x5 kernel
            cv::cvtColor(frame_blurred, hsv, cv::COLOR_BGR2HSV); // convert entire frame to hsv colour space
            // TODO: need to know when
            // 1. hand is present or absent by checking mediapipe data 
            // 2. when only frame info is available (i.e. MP at 10 FPS and camera at 30 FPS, there will be times when we only have frame data)
            if (landmarks_available && rect_available) { // media pipe output available as well as frame info
                hsv_roi = hsv(sample_region); // for histogram adaption
                hand_hist = adapt_histogram(hsv_roi, hand_hist); // adapt the histogram
                /* Update where we are */
                prediction = hand_tracker.update_position(hsv, bbox, true); // true since MP rect is available
                hand_tracker.new_hand.update_velocity(hand_tracker.old_hand);
                hand_tracker.old_hand.set_state(hand_tracker.new_hand.centroid, hand_tracker.new_hand.bound, hand_tracker.new_hand.area);
                hand_tracker.old_hand.velocity = hand_tracker.new_hand.velocity;
            } else { // only frame information available
                prediction = hand_tracker.update_position(hsv, bbox, false); // pass an empty rect
                hand_tracker.new_hand.update_velocity(hand_tracker.old_hand);
                hand_tracker.old_hand.set_state(hand_tracker.new_hand.centroid, hand_tracker.new_hand.bound, hand_tracker.new_hand.area);
                hand_tracker.old_hand.velocity = hand_tracker.new_hand.velocity;
            }
            auto output_rects = absl::make_unique<std::vector<Rect>>(num_rects);
            auto output_landmark = absl::make_unique<Landmark>();
            
            if (!landmarks_available) {
                convert_bbox(hand_tracker.new_hand.bound, &(output_rects->at(0)));
            } else {
                convert_bbox(hand_tracker.new_hand.bound, &(output_rects->at(0)));
                convert_bbox(sample_region, &(output_rects->at(1)));
            }

            convert_centroid(hand_tracker.new_hand.centroid, output_landmark.get());
            cc->Outputs().Tag(kRectsTag).Add(output_rects.release(), cc->InputTimestamp());
            cc->Outputs().Tag(kLandmarkTag).Add(output_landmark.release(), cc->InputTimestamp());
            // cout << "Hand Vel: " << hand_tracker.new_hand.centroid << endl;
            // cv::rectangle(frame_blurred, hand_tracker.new_hand.bound, red, 2);
            // cv::rectangle(frame_blurred, prediction, green, 2);
            // cv::rectangle(frame_blurred, sample_region, blue, 2);
            // cv::circle(frame_blurred, hand_tracker.new_hand.centroid, 5, blue, -1);
            // cv::imshow("Feed", frame_blurred);
        } 

        forwarder->send(data.c_str(), data.length());

        return(::mediapipe::OkStatus());
    }

    ::mediapipe::Status Close(CalculatorContext* cc) override
    {
        delete forwarder;

        return(::mediapipe::OkStatus());
    }

    cv::Rect get_sample_rect(const CalculatorContext* cc, int src_width, int src_height) {
        /*
        Inputs:
            cc: calculator context
            src_width: width of the source frame
            src_height: height of the source frame
        Outputs:
            rect: a CV rect that contains the coordinates of the region to sample for the hand histogram
        */
        const auto& landmarks = cc->Inputs().Tag(kLandmarksTag).Get<NormalizedLandmarkList>();

        int keypoints[] = {0, 1, 5, 9, 13, 17}; 
        int size = sizeof(keypoints) / sizeof(keypoints[0]);
        float center_x = 0;
        float center_y = 0;

        for(int i = 0; i < size; i++)
        {
            const NormalizedLandmark& keypoint = landmarks.landmark(keypoints[i]);
            center_x += keypoint.x()*src_width;
            center_y += keypoint.y()*src_height;
        }

        center_x /= size;
        center_y /= size;

        // create rectangle
        cv::Point top_left (clip(center_x - SAMPLE_BOX_SIZE/2, 0, src_width), clip(center_y - SAMPLE_BOX_SIZE/2, 0, src_height));
        cv::Point bottom_right (clip(center_x + SAMPLE_BOX_SIZE/2, 0, src_width), clip(center_y + SAMPLE_BOX_SIZE/2, 0, src_height));
        cv::Rect roi (top_left, bottom_right);

        return roi;
    }

    cv::Rect get_bounding_box(const CalculatorContext* cc, int src_width, int src_height) {
        /*
        Inputs:
            cc: calculator context
            src_width: width of the source frame
            src_height: height of the source frame
        Outputs:
            rect: a CV rect that will represent a bounding box for the hand in the frame by manipulating the normalized rect
        */

        // by default use the whole frame to search
        int width = src_width-1;
        int height = src_height-1;
        int min_x = 1;
        int min_y = 1;

        const auto& norm_rect = cc->Inputs().Tag(kNormRectTag).Get<NormalizedRect>();

        if (norm_rect.width() > 0.0 && norm_rect.height() > 0.0) {
            float normalized_width = norm_rect.width();
            float normalized_height = norm_rect.height();
            int x_center = std::round(norm_rect.x_center() * src_width);
            int y_center = std::round(norm_rect.y_center() * src_height);
            width = std::round(normalized_width * src_width);
            height = std::round(normalized_height * src_height);
            min_x = std::round(x_center - width/2); 
            min_y = std::round(y_center - height/2); 
            min_x = clip(min_x, 0, src_width-width);
            min_y = clip(min_y, 0, src_height-height);
        } 
        
        cv::Rect roi (min_x, min_y, width, height);
        return roi;
    }

    void convert_bbox(cv::Rect bounding_box, Rect* rect) {
        rect->set_x_center(bounding_box.x + bounding_box.width / 2);
        rect->set_y_center(bounding_box.y + bounding_box.height / 2);
        rect->set_width(bounding_box.width);
        rect->set_height(bounding_box.height);
    }

    void convert_centroid(cv::Point centroid, Landmark* landmark) {
        landmark->set_x(centroid.x);
        landmark->set_y(centroid.y);
    }

};

REGISTER_CALCULATOR(HandTrackingCalculator);
}

