#include <unistd.h>
#include <string.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/udp.h"

namespace mediapipe {

namespace {
constexpr char kLandmarksTag[] = "LANDMARKS";
}

const char *GL_IP = "localhost";    // IP address of gesture learning UDP server
const short GL_PORT = 2000;         // port number of gesture learning UDP server

const char *MC_IP = "localhost";    // IP address of mouse control UDP server
const short MC_PORT = 3000;         // port number of mouse control UDP server

class LandmarkForwarderCalculator : public CalculatorBase {
private:
    udp::client *gl_forwarder;
    udp::client *mc_forwarder;

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
        gl_forwarder = new udp::client(GL_IP, GL_PORT);
        mc_forwarder = new udp::client(MC_IP, MC_PORT);

        return(::mediapipe::OkStatus());
    }

    ::mediapipe::Status Process(CalculatorContext* cc) override
    {
        if(cc->Inputs().Tag(kLandmarksTag).IsEmpty())
        {
            return(::mediapipe::OkStatus());
        }

        const auto& landmarks = cc->Inputs().Tag(kLandmarksTag).Get<NormalizedLandmarkList>();
        std::string data;

        for(int i = 0; i < landmarks.landmark_size(); i++)
        {
            const NormalizedLandmark& landmark = landmarks.landmark(i);

            data += std::to_string(landmark.x()) + ',';
            data += std::to_string(landmark.y()) + ',';
            data += std::to_string(landmark.z()) + ';';
        }

        gl_forwarder->send(data.c_str(), data.length());
        mc_forwarder->send(data.c_str(), data.length());

        return(::mediapipe::OkStatus());
    }

    ::mediapipe::Status Close(CalculatorContext* cc) override
    {
        delete gl_forwarder;
        delete mc_forwarder;

        return(::mediapipe::OkStatus());
    }
};

REGISTER_CALCULATOR(LandmarkForwarderCalculator);
}