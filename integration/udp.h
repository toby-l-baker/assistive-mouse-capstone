#ifndef UDP_H
#define UDP_H

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdexcept>

namespace udp {


// =============================== CLIENT ===================================

class client
{
public:
    client(const std::string& addr, int port);
    ~client();

    int get_socket() const;
    int get_port() const;
    std::string get_addr() const;
    int send(const char *msg, size_t size);

private:
    int f_socket;
    int f_port;
    std::string f_addr;
    struct addrinfo *f_addrinfo;
};


// =============================== SERVER ===================================

class server
{
public:
    server(const std::string& addr, int port);
    ~server();

    int get_socket() const;
    int get_port() const;
    std::string get_addr() const;
    int recv(char *msg, size_t max_size);
    int timed_recv(char *msg, size_t max_size, int max_wait_ms);

private:
    int f_socket;
    int f_port;
    std::string f_addr;
    struct addrinfo *f_addrinfo;
};

}
#endif
