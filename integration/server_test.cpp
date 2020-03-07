#include <stdio.h>
#include <string.h>
#include "udp.h"

using namespace udp;

int main(int argc, char **argv)
{
    char msg[256];
    int data;

    memset(&msg, '\0', sizeof(msg));

    printf("[server_test]: creating server\n");

    server udp_server("127.0.0.1", 7777);
    
    printf("[server_test]: server created\n");
    printf("[server_test]: server address = %s\n", udp_server.get_addr().c_str());
    printf("[server_test]: server port = %d\n", udp_server.get_port());
    printf("[server_test]: waiting for data\n");
    
    data = udp_server.recv(msg, sizeof(msg));
    
    printf("[server_test]: recieved %d bytes\n", data);
    printf("[server_test]: [%s]\n", msg);

    return(0);
}

