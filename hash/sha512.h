#ifndef SHA512_H
#define SHA512_H
#include <string>

void sha512(unsigned char *input, int length, unsigned char *digest);
void pbkdf2_hmac_sha512(uint8_t *out, size_t outlen,const uint8_t *passwd, size_t passlen,const uint8_t *salt, size_t saltlen,uint64_t iter);
void hmac_sha512(unsigned char *key, int key_length, unsigned char *message, int message_length, unsigned char *digest);

std::string sha512_hex(unsigned char *digest);

#endif
