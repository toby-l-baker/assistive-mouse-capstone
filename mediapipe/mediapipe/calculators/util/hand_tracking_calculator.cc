#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/udp.h"

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
#define SAMPLE_BOX_SIZE 25

/*** GLOBALS ***/

// width and height of the frame
double dHeight;
double dWidth;

// For drawing on frames
cv::Scalar blue (255, 0, 0);
cv::Scalar green (0, 255, 0);
cv::Scalar red (0, 0, 255);

// Hand Histogram
cv::Mat hand_hist;

// Setup for various histogram values
int ch[] = {0, 1};
int hist_size[] = {H_BINS, S_BINS};
float h_ranges[] = {0, 180};
float s_ranges[] = {0, 255};
const float* ranges[] = { h_ranges, s_ranges };

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
        void update_position(cv::Mat, cv::Rect roi_mp, bool mp_available); // locates the hand in the bounded region defined by predict_position
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
void HandTracker::update_position(cv::Mat frame, cv::Rect roi_mp, bool mp_available) {

    cv::Rect roi (roi_mp.x, roi_mp.y, roi_mp.width, roi_mp.height);
    
    if (mp_available == false) {
        roi = predict_position();
    }

    vector<vector<Point>> contours = get_contours(frame(roi)); // this will be in local coords

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
}

namespace mediapipe {

namespace {
constexpr char kInputFrameTag[] = "IMAGE";
constexpr char kLandmarksTag[] = "LANDMARKS";
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
    // no need for VideoCapture(0); or cap.set(cv::CAP_PROP_FPS, 30);

    // no need for dWidth or dHeight, use MP keypoints for sampling/initializing

    // Hand Histogram
    cv::Mat hand_hist;

    // Setup for various histogram values
    int ch[2];
    int hist_size[2];
    float h_ranges[2];
    float s_ranges[2];
    const float* ranges[2];

public:
    static ::mediapipe::Status GetContract(CalculatorContract* cc)
    {
        RET_CHECK(cc->Inputs().HasTag(kLandmarksTag)) << "No input stream provided.";

        if(cc->Inputs().HasTag(kLandmarksTag))
        {
            cc->Inputs().Tag(kLandmarksTag).Set<NormalizedLandmarkList>();
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

        ch[0] = 0;
        ch[1] = 1;
        hist_size[0] = H_BINS;
        hist_size[1] = S_BINS;
        h_ranges[0] = 0;
        h_ranges[1] = 180;
        s_ranges[0] = 0;
        s_ranges[1] = 255;
        ranges[0] = h_ranges;
        ranges[1] = s_ranges;

        calibrated = false; // uncalibrated histogram initially

        return(::mediapipe::OkStatus());
    }

    ::mediapipe::Status Process(CalculatorContext* cc) override
    {
        string data;

        if(cc->Inputs().Tag(kLandmarksTag).IsEmpty())
        {
            return(::mediapipe::OkStatus());
        }

        const auto& landmarks = cc->Inputs().Tag(kLandmarksTag).Get<NormalizedLandmarkList>();

        int keypoints[] = {0, 1, 5, 9, 13, 17}; 
        int size = sizeof(keypoints) / sizeof(keypoints[0]);
        float center_x = 0;
        float center_y = 0;

        for(int i = 0; i < size; i++)
        {
            const NormalizedLandmark& keypoint = landmarks.landmark(keypoints[i]);
            center_x += keypoint.x();
            center_y += keypoint.y();
        }

        center_x /= size;
        center_y /= size;

        /*
        if (calibrated == false) {          
            frame = NULL;
            frame = frame_from_input; // TODO: convert from input image to cv::Mat
            cv::GaussianBlur(frame, frame_blurred, Size(BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0); // Blur image with 5x5 kernel
            cv::Rect roi_rect (top_left_x, top_left_y, BOX_SIZE, BOX_SIZE);  // TODO: need to get top_left_x and top_left_y from MP keypoints around the palm to sample a rectangle in the middle of the hand
            cv::Mat roi = frame_blurred(roi_rect) // grab the region of the frame we want
            cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV); // convert roi to HSV colour space
            hand_hist = get_histogram(hsv_roi); // get initial histogram

            calibrated = true; // hand histogram has been calibrated

            // Currently uses largest contour by area to initialise the hand
            cv::cvtColor(frame_blurred, hsv, cv::COLOR_BGR2HSV); // convert entire frame to hsv colour space
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
            hand_tracker.old_hand.set_state(centroid, bound, max_area);
            hand_tracker.old_hand.velocity = {0, 0};
        } else {
            frame = frame_from_mediapipe; // TODO
            cv::GaussianBlur(frame, frame_blurred, Size(BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0); // Blur image with 5x5 kernel
            cv::cvtColor(frame_blurred, hsv, cv::COLOR_BGR2HSV); // convert entire frame to hsv colour space
            // TODO: need to know when
            // 1. hand is present or absent by checking mediapipe data 
            // 2. when only frame info is available (i.e. MP at 10 FPS and camera at 30 FPS, there will be times when we only have frame data)
            if (mediapipe_data) { // media pipe output available as well as frame info
                if (hand_present) { // hand is in the frame 
                    cv::Rect roi_rect (top_left_x, top_left_y, SAMPLE_BOX_SIZE, SAMPLE_BOX_SIZE);  // TODO: need to get top_left_x and top_left_y from MP keypoints around the palm to sample a rectangle in the middle of the hand
                    hsv_roi = hsv(roi_rect); // for histogram adaption
                    hand_hist = adapt_histogram(hsv_roi, hand_hist); // adapt the histogram
                    cv::Rect roi_mp (top_left_hand_y, top_left_hand_y, hand_width, hand_height); // TODO: unpack MP bbox info to initialize CV rect for hand bounding box
                    hand_tracker.update_position(hsv, roi_mp, true); // true since MP rect is available
                    hand_tracker.new_hand.update_velocity(hand_tracker.old_hand);
                    hand_tracker.old_hand.set_state(hand_tracker.new_hand.centroid, hand_tracker.new_hand.bound, hand_tracker.new_hand.area);
                    hand_tracker.old_hand.velocity = hand_tracker.new_hand.velocity;
                } else { // hand is not in the frame will need to recalibrate once it re-enters
                    calibrated = false;
                }
            } else { // only frame information available
                hand_tracker.update_position(hsv, cv::Rect roi_mp (0, 0, 1, 1), false); // pass an empty rect
                hand_tracker.new_hand.update_velocity(hand_tracker.old_hand);
                hand_tracker.old_hand.set_state(hand_tracker.new_hand.centroid, hand_tracker.new_hand.bound, hand_tracker.new_hand.area);
                hand_tracker.old_hand.velocity = hand_tracker.new_hand.velocity;
            }
        }
        */
        
        forwarder->send(data.c_str(), data.length());

        return(::mediapipe::OkStatus());
    }

    ::mediapipe::Status Close(CalculatorContext* cc) override
    {
        delete forwarder;

        return(::mediapipe::OkStatus());
    }
};

REGISTER_CALCULATOR(HandTrackingCalculator);
}

