/* Bring the loopback interface up with 127.0.0.1/8 — busybox here has no
 * ifconfig/ip applet, and lo starts DOWN, so localhost is unreachable. */
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <netinet/in.h>
#include <string.h>

int main(void) {
    int s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s < 0) return 1;
    struct ifreq ifr;
    memset(&ifr, 0, sizeof ifr);
    strcpy(ifr.ifr_name, "lo");
    struct sockaddr_in *a = (void *)&ifr.ifr_addr;
    a->sin_family = AF_INET;
    a->sin_addr.s_addr = htonl(0x7f000001);     /* 127.0.0.1 */
    ioctl(s, SIOCSIFADDR, &ifr);
    a->sin_addr.s_addr = htonl(0xff000000);     /* 255.0.0.0 */
    ioctl(s, SIOCSIFNETMASK, &ifr);
    ioctl(s, SIOCGIFFLAGS, &ifr);
    ifr.ifr_flags |= IFF_UP | IFF_RUNNING;
    ioctl(s, SIOCSIFFLAGS, &ifr);
    return 0;
}
