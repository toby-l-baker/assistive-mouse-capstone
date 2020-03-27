//Uncomment the following line if you are compiling this code in Visual Studio
//#include "stdafx.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "params.hpp"

using namespace cv;
using namespace std;

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
        void update_position(cv::Mat); // locates the hand in the bounded region defined by predict_position
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
void HandTracker::update_position(cv::Mat frame) {
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV); // convert the frame to hsv colour space
    cv::Rect roi = predict_position();

    vector<vector<Point>> contours = get_contours(hsv(roi)); // this will be in local coords

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

    // Update the hand histogram
    // TODO: replace with something that uses the keypoints of mediapipe e.g. centre of palm or use a small square around each keypoint
    cv::Rect hand_sample (centroid.x - SAMPLE_BOX_SIZE/2, centroid.y - SAMPLE_BOX_SIZE/2, SAMPLE_BOX_SIZE, SAMPLE_BOX_SIZE);
    hand_hist = adapt_histogram(hsv(hand_sample), hand_hist);

    // Set the state of the new hand
    new_hand.set_state(centroid, bound, max_area);

    // Draw the search region and hand location on the frame
    cv::rectangle(frame, bound, red, 2);
    cv::rectangle(frame, roi, green, 2);
    // cv::rectangle(frame, hand_sample, blue, 2);
    cv::circle(frame, centroid, 5, green, -1);
}

int main(int argc, char* argv[])
{
    //Open the default video camera
    VideoCapture cap(0);

    // if not success, exit program
    if (cap.isOpened() == false)  
    {
    cout << "Cannot open the video camera" << endl;
    cin.get(); //wait for any key press
    return -1;
    } 

    cap.set(cv::CAP_PROP_FPS, 30); // set frame rate to 30

    bool calibrated = false;

    HandTracker hand_tracker;

    cv::Mat frame;
    cv::Mat frame_blurred;

    while (true)
    {
        bool bSuccess = cap.read(frame); // read a new frame from video 
        cv::GaussianBlur(frame, frame_blurred, Size(BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0); // Blur image with 5x5 kernel

        // HISTOGRAM CALIBRATION
        // TODO: interface with medaipipe so each time the camera moves from our of frame to in frame we take a sample and initialise the hands position
        while (calibrated == false) {
            bSuccess = cap.read(frame); // read a new frame from video 
            cv::GaussianBlur(frame, frame_blurred, Size(BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0); // Blur image with 5x5 kernel
            dWidth = frame.size().width;
            dHeight = frame.size().height;
            int centre [2] = {dWidth/2, dHeight/2}; // centre of the frame
            cv::Rect roi_rect (centre[0], centre[1], BOX_SIZE, BOX_SIZE);
            cv::rectangle(frame_blurred, roi_rect.tl(), roi_rect.br(), blue);
            if (waitKey(10) == 'z') { // hand is in the right spot, sample histogram
                cv::Mat roi = frame_blurred(roi_rect);
                cv::Mat hsv;
                cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
                hand_hist = get_histogram(hsv);
                calibrated = true;
                cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
                vector<vector<Point>> conts = get_contours(hsv);

                // Currently uses largest contour by area
                double max_area = 0.0;
                vector<Point> max_contour;
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
                cv::destroyWindow("Calibration Feed");
                break;
            }
            cv::imshow("Calibration Feed", frame_blurred);
        }

        // Main functionality

        hand_tracker.update_position(frame_blurred);
        hand_tracker.new_hand.update_velocity(hand_tracker.old_hand);
        hand_tracker.old_hand.set_state(hand_tracker.new_hand.centroid, hand_tracker.new_hand.bound, hand_tracker.new_hand.area);
        hand_tracker.old_hand.velocity = hand_tracker.new_hand.velocity;


        //show the filtered back projection
        cv::imshow("Camera Feed", frame_blurred);

        // Breaking the while loop if the frames cannot be captured
        if (bSuccess == false) {break;}

        // Check if 'esc' is pressed
        if (waitKey(10) == 27) {break;}
    }

 return 0;

}