#include <unistd.h>
#include <string.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/ret_check.h"

namespace udp {
class client
{
private:
    int f_socket;
    int f_port;
    std::string f_addr;
    struct addrinfo *f_addrinfo;

public:
    client(const std::string& addr, int port) : f_port(port), f_addr(addr)
    {
        char decimal_port[16];
        struct addrinfo hints;
        int ret_val;

        snprintf(decimal_port, sizeof(decimal_port), "%d", f_port);

        memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_DGRAM;
        hints.ai_protocol = IPPROTO_UDP;

        ret_val = getaddrinfo(f_addr.c_str(), decimal_port, &hints, &f_addrinfo);

        if(ret_val != 0 || f_addrinfo == NULL)
        {
            throw std::runtime_error(("invalid address or port: \"" + addr + ":" + decimal_port + "\"").c_str());
        }

        f_socket = socket(f_addrinfo->ai_family, SOCK_DGRAM | SOCK_CLOEXEC, IPPROTO_UDP);

        if(f_socket == -1)
        {
            freeaddrinfo(f_addrinfo);
            throw std::runtime_error(("could not create socket for: \"" + addr + ":" + decimal_port + "\"").c_str());
        }
    }

    ~client()
    {
        freeaddrinfo(f_addrinfo);
        close(f_socket);
    }

    int get_socket() const
    {
        return(f_socket);
    }

    int get_port() const
    {
        return(f_port);
    }

    std::string get_addr() const
    {
        return(f_addr);
    }

    int send(const char *msg, size_t size)
    {
        return(sendto(f_socket, msg, size, 0, f_addrinfo->ai_addr, f_addrinfo->ai_addrlen));
    }
};
}


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