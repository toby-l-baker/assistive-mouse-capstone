#include <unistd.h>
#include <string.h>
#include "udp.h"

namespace udp {


// =============================== CLIENT ===================================

client::client(const std::string& addr, int port) : f_port(port), f_addr(addr)
{
    char decimal_port[16];
    snprintf(decimal_port, sizeof(decimal_port), "%d", f_port);

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_protocol = IPPROTO_UDP;

    int r(getaddrinfo(addr.c_str(), decimal_port, &hints, &f_addrinfo));

    if(r != 0 || f_addrinfo == NULL)
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

client::~client()
{
    freeaddrinfo(f_addrinfo);
    close(f_socket);
}

int client::get_socket() const
{
    return(f_socket);
}

int client::get_port() const
{
    return(f_port);
}

std::string client::get_addr() const
{
    return(f_addr);
}


int client::send(const char *msg, size_t size)
{
    return(sendto(f_socket, msg, size, 0, f_addrinfo->ai_addr, f_addrinfo->ai_addrlen));
}


// =============================== SERVER ===================================

server::server(const std::string& addr, int port) : f_port(port), f_addr(addr)
{
    char decimal_port[16];
    snprintf(decimal_port, sizeof(decimal_port), "%d", f_port);
    decimal_port[sizeof(decimal_port) / sizeof(decimal_port[0] - 1)] = '\0';
    
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_protocol = IPPROTO_UDP;

    int r(getaddrinfo(addr.c_str(), decimal_port, &hints, &f_addrinfo));
    
    if(r != 0 || f_addrinfo == NULL)
    {
        throw std::runtime_error(("invalid address or port for UDP socket: \"" + addr + ":" + decimal_port + "\"").c_str());
    }

    f_socket = socket(f_addrinfo->ai_family, SOCK_DGRAM | SOCK_CLOEXEC, IPPROTO_UDP);

    if(f_socket == -1)
    {
        freeaddrinfo(f_addrinfo);
        throw std::runtime_error(("could not create UDP socket for: \"" + addr + ":" + decimal_port + "\"").c_str());
    }

    r = bind(f_socket, f_addrinfo->ai_addr, f_addrinfo->ai_addrlen);

    if(r != 0)
    {
        freeaddrinfo(f_addrinfo);
        close(f_socket);
        throw std::runtime_error(("could not bind UDP socket with: \"" + addr + ":" + decimal_port + "\"").c_str());
    }
}

server::~server()
{
    freeaddrinfo(f_addrinfo);
    close(f_socket);
}

int server::get_socket() const
{
    return(f_socket);
}

int server::get_port() const
{
    return(f_port);
}

std::string server::get_addr() const
{
    return(f_addr);
}

int server::recv(char *msg, size_t max_size)
{
    return(::recv(f_socket, msg, max_size, 0));
}

int server::timed_recv(char *msg, size_t max_size, int max_wait_ms)
{
    fd_set s;
    FD_ZERO(&s);
    FD_SET(f_socket, &s);
    
    struct timeval timeout;
    timeout.tv_sec = max_wait_ms / 1000;
    timeout.tv_usec = (max_wait_ms % 1000) * 1000;

    int retval = select(f_socket + 1, &s, &s, &s, &timeout);

    if(retval == -1)
    {
        return(-1);
    }
    else if(retval > 0)
    {
        return(::recv(f_socket, msg, max_size, 0));
    }
    else
    {
        errno = EAGAIN;
        return(-1);
    }    
}

}
