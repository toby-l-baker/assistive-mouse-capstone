#include <stdio.h>
#include <string.h>
#include "udp.h"

using namespace udp;

int main(int argc, char **argv)
{
    const char *msg = "hello world";
    int data;

    printf("[client_test]: creating client\n");

    client udp_client("127.0.0.1", 7777);
    
    printf("[client_test]: client created\n");
    printf("[client_test]: client address = %s\n", udp_client.get_addr().c_str());
    printf("[client_test]: client port = %d\n", udp_client.get_port());
    printf("[client_test]: sending data [%s]\n", msg);
    
    data = udp_client.send(msg, strlen(msg));
    
    printf("[client_test]: sent %d bytes\n", data);

    return(0);
}

