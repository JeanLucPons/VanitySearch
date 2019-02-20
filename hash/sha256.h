#ifndef SHA256_H
#define SHA256_H
#include <string>
 
void sha256(uint8_t *input,int length, uint8_t *digest);
void sha256_33(uint8_t *input, uint8_t *digest);
void sha256sse_33(uint8_t *i0, uint8_t *i1, uint8_t *i2, uint8_t *i3,
  uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3);
void sha256sse_65(uint8_t *i0, uint8_t *i1, uint8_t *i2, uint8_t *i3,
  uint8_t *d0, uint8_t *d1, uint8_t *d2, uint8_t *d3);
std::string sha256_hex(unsigned char *digest);
void sha256sse_test();

#endif