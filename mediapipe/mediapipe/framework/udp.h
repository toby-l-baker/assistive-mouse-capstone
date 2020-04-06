#ifndef MEDIAPIPE_FRAMEWORK_UDP_H_
#define MEDIAPIPE_FRAMEWORK_UDP_H_

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

namespace mediapipe {
namespace udp {

// =============================== CLIENT ===================================

class client
{
private:
    int f_socket;
    int f_port;
    std::string f_addr;
    struct addrinfo *f_addrinfo;

public:
    client(const std::string& addr, int port);
    ~client();

    int get_socket() const;
    int get_port() const;
    std::string get_addr() const;
    int send(const char *msg, size_t size);
};

// =============================== SERVER ===================================
class server
{
private:
    int f_socket;
    int f_port;
    std::string f_addr;
    struct addrinfo *f_addrinfo;

public:
    server(const std::string& addr, int port);
    ~server();

    int get_socket() const;
    int get_port() const;
    std::string get_addr() const;
    int recv(char *msg, size_t max_size);
    int timed_recv(char *msg, size_t max_size, int max_wait_ms);
};

}
}
#endif