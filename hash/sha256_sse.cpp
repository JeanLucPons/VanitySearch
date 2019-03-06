/*
 * This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2019 Jean Luc PONS.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "sha256.h"
#include <immintrin.h>
#include <string.h>
#include <stdint.h>

#define BSWAP

namespace _sha256sse
{

#ifndef WIN64
#define _byteswap_ulong __builtin_bswap32
#define _byteswap_uint64 __builtin_bswap64
#endif

#ifdef BSWAP
#define WRITEBE32(ptr,xPtr,o) *((uint32_t *)(ptr)) = _byteswap_ulong(*((uint32_t *)(xPtr)+o))
#define WRITEBE64(ptr,x) *((uint64_t *)(ptr)) = _byteswap_uint64(x)
#define READBE32(ptr) (uint32_t)_byteswap_ulong(*(uint32_t *)(ptr))
#else
#define WRITEBE32(ptr,x) *(ptr) = x
#define WRITEBE64(ptr,x) *(ptr) = x
#define READBE32(ptr) *(uint32_t *)(ptr)
#endif

#ifdef WIN64
  static const __declspec(align(16)) uint32_t _init[] = {
#else
  static const uint32_t _init[] __attribute__ ((aligned (16))) = {
#endif
      0x6a09e667,0x6a09e667,0x6a09e667,0x6a09e667,
      0xbb67ae85,0xbb67ae85,0xbb67ae85,0xbb67ae85,
      0x3c6ef372,0x3c6ef372,0x3c6ef372,0x3c6ef372,
      0xa54ff53a,0xa54ff53a,0xa54ff53a,0xa54ff53a,
      0x510e527f,0x510e527f,0x510e527f,0x510e527f,
      0x9b05688c,0x9b05688c,0x9b05688c,0x9b05688c,
      0x1f83d9ab,0x1f83d9ab,0x1f83d9ab,0x1f83d9ab,
      0x5be0cd19,0x5be0cd19,0x5be0cd19,0x5be0cd19
  };

//#define Maj(x,y,z) ((x&y)^(x&z)^(y&z))
//#define Ch(x,y,z)  ((x&y)^(~x&z))

// The following functions are equivalent to the above
//#define Maj(x,y,z) ((x & y) | (z & (x | y)))
//#define Ch(x,y,z) (z ^ (x & (y ^ z)))

#define Maj(b,c,d) _mm_or_si128(_mm_and_si128(b, c), _mm_and_si128(d, _mm_or_si128(b, c)) )
#define Ch(b,c,d)  _mm_xor_si128(_mm_and_si128(b, c) , _mm_andnot_si128(b , d) )
#define ROR(x,n)   _mm_or_si128( _mm_srli_epi32(x, n) , _mm_slli_epi32(x, 32 - n) )
#define SHR(x,n)   _mm_srli_epi32(x, n)

  /* SHA256 Functions */
#define	S0(x) (_mm_xor_si128(ROR((x), 2) , _mm_xor_si128(ROR((x), 13), ROR((x), 22))))
#define	S1(x) (_mm_xor_si128(ROR((x), 6) , _mm_xor_si128(ROR((x), 11), ROR((x), 25))))
#define	s0(x) (_mm_xor_si128(ROR((x), 7) , _mm_xor_si128(ROR((x), 18), SHR((x), 3))))
#define	s1(x) (_mm_xor_si128(ROR((x), 17), _mm_xor_si128(ROR((x), 19), SHR((x), 10))))

#define add4(x0, x1, x2, x3) _mm_add_epi32(_mm_add_epi32(x0, x1), _mm_add_epi32(x2, x3))
#define add3(x0, x1, x2 ) _mm_add_epi32(_mm_add_epi32(x0, x1), x2)
#define add5(x0, x1, x2, x3, x4) _mm_add_epi32(add3(x0, x1, x2), _mm_add_epi32(x3, x4))


#define	Round(a, b, c, d, e, f, g, h, i, w)                 \
    T1 = add5(h, S1(e), Ch(e, f, g), _mm_set1_epi32(i), w);	\
    d = _mm_add_epi32(d, T1);                               \
    T2 = _mm_add_epi32(S0(a), Maj(a, b, c));                \
    h = _mm_add_epi32(T1, T2);

  #define LOADW(i) \
    _mm_set_epi32(READBE32(blk[0] + i * 4), \
                  READBE32(blk[1] + i * 4), \
                  READBE32(blk[2] + i * 4), \
                  READBE32(blk[3] + i * 4));

#define WMIX() \
  w0 = add4(s1(w14), w9, s0(w1), w0); \
  w1 = add4(s1(w15), w10, s0(w2), w1); \
  w2 = add4(s1(w0), w11, s0(w3), w2); \
  w3 = add4(s1(w1), w12, s0(w4), w3); \
  w4 = add4(s1(w2), w13, s0(w5), w4); \
  w5 = add4(s1(w3), w14, s0(w6), w5); \
  w6 = add4(s1(w4), w15, s0(w7), w6); \
  w7 = add4(s1(w5), w0, s0(w8), w7); \
  w8 = add4(s1(w6), w1, s0(w9), w8); \
  w9 = add4(s1(w7), w2, s0(w10), w9); \
  w10 = add4(s1(w8), w3, s0(w11), w10); \
  w11 = add4(s1(w9), w4, s0(w12), w11); \
  w12 = add4(s1(w10), w5, s0(w13), w12); \
  w13 = add4(s1(w11), w6, s0(w14), w13); \
  w14 = add4(s1(w12), w7, s0(w15), w14); \
  w15 = add4(s1(w13), w8, s0(w0), w15);

  // Initialise state
  void Initialize(__m128i *s) {
    memcpy(s, _init, sizeof(_init));
  }

  // Perform 4 SHA in parallel using SSE2
  void Transform(__m128i *s, uint8_t *blk[4])
  {
    __m128i a,b,c,d,e,f,g,h;
    __m128i w0, w1, w2, w3, w4, w5, w6, w7;
    __m128i w8, w9, w10, w11, w12, w13, w14, w15;
    __m128i T1, T2;

    a = _mm_load_si128(s + 0);
    b = _mm_load_si128(s + 1);
    c = _mm_load_si128(s + 2);
    d = _mm_load_si128(s + 3);
    e = _mm_load_si128(s + 4);
    f = _mm_load_si128(s + 5);
    g = _mm_load_si128(s + 6);
    h = _mm_load_si128(s + 7);

    w0 = LOADW(0);
    w1 = LOADW(1);
    w2 = LOADW(2);
    w3 = LOADW(3);
    w4 = LOADW(4);
    w5 = LOADW(5);
    w6 = LOADW(6);
    w7 = LOADW(7);
    w8 = LOADW(8);
    w9 = LOADW(9);
    w10 = LOADW(10);
    w11 = LOADW(11);
    w12 = LOADW(12);
    w13 = LOADW(13);
    w14 = LOADW(14);
    w15 = LOADW(15);

    Round(a, b, c, d, e, f, g, h, 0x428A2F98, w0);
    Round(h, a, b, c, d, e, f, g, 0x71374491, w1);
    Round(g, h, a, b, c, d, e, f, 0xB5C0FBCF, w2);
    Round(f, g, h, a, b, c, d, e, 0xE9B5DBA5, w3);
    Round(e, f, g, h, a, b, c, d, 0x3956C25B, w4);
    Round(d, e, f, g, h, a, b, c, 0x59F111F1, w5);
    Round(c, d, e, f, g, h, a, b, 0x923F82A4, w6);
    Round(b, c, d, e, f, g, h, a, 0xAB1C5ED5, w7);
    Round(a, b, c, d, e, f, g, h, 0xD807AA98, w8);
    Round(h, a, b, c, d, e, f, g, 0x12835B01, w9);
    Round(g, h, a, b, c, d, e, f, 0x243185BE, w10);
    Round(f, g, h, a, b, c, d, e, 0x550C7DC3, w11);
    Round(e, f, g, h, a, b, c, d, 0x72BE5D74, w12);
    Round(d, e, f, g, h, a, b, c, 0x80DEB1FE, w13);
    Round(c, d, e, f, g, h, a, b, 0x9BDC06A7, w14);
    Round(b, c, d, e, f, g, h, a, 0xC19BF174, w15);

    WMIX()

    Round(a, b, c, d, e, f, g, h, 0xE49B69C1, w0);
    Round(h, a, b, c, d, e, f, g, 0xEFBE4786, w1);
    Round(g, h, a, b, c, d, e, f, 0x0FC19DC6, w2);
    Round(f, g, h, a, b, c, d, e, 0x240CA1CC, w3);
    Round(e, f, g, h, a, b, c, d, 0x2DE92C6F, w4);
    Round(d, e, f, g, h, a, b, c, 0x4A7484AA, w5);
    Round(c, d, e, f, g, h, a, b, 0x5CB0A9DC, w6);
    Round(b, c, d, e, f, g, h, a, 0x76F988DA, w7);
    Round(a, b, c, d, e, f, g, h, 0x983E5152, w8);
    Round(h, a, b, c, d, e, f, g, 0xA831C66D, w9);
    Round(g, h, a, b, c, d, e, f, 0xB00327C8, w10);
    Round(f, g, h, a, b, c, d, e, 0xBF597FC7, w11);
    Round(e, f, g, h, a, b, c, d, 0xC6E00BF3, w12);
    Round(d, e, f, g, h, a, b, c, 0xD5A79147, w13);
    Round(c, d, e, f, g, h, a, b, 0x06CA6351, w14);
    Round(b, c, d, e, f, g, h, a, 0x14292967, w15);

    WMIX()

    Round(a, b, c, d, e, f, g, h, 0x27B70A85, w0);
    Round(h, a, b, c, d, e, f, g, 0x2E1B2138, w1);
    Round(g, h, a, b, c, d, e, f, 0x4D2C6DFC, w2);
    Round(f, g, h, a, b, c, d, e, 0x53380D13, w3);
    Round(e, f, g, h, a, b, c, d, 0x650A7354, w4);
    Round(d, e, f, g, h, a, b, c, 0x766A0ABB, w5);
    Round(c, d, e, f, g, h, a, b, 0x81C2C92E, w6);
    Round(b, c, d, e, f, g, h, a, 0x92722C85, w7);
    Round(a, b, c, d, e, f, g, h, 0xA2BFE8A1, w8);
    Round(h, a, b, c, d, e, f, g, 0xA81A664B, w9);
    Round(g, h, a, b, c, d, e, f, 0xC24B8B70, w10);
    Round(f, g, h, a, b, c, d, e, 0xC76C51A3, w11);
    Round(e, f, g, h, a, b, c, d, 0xD192E819, w12);
    Round(d, e, f, g, h, a, b, c, 0xD6990624, w13);
    Round(c, d, e, f, g, h, a, b, 0xF40E3585, w14);
    Round(b, c, d, e, f, g, h, a, 0x106AA070, w15);

    WMIX()

    Round(a, b, c, d, e, f, g, h, 0x19A4C116, w0);
    Round(h, a, b, c, d, e, f, g, 0x1E376C08, w1);
    Round(g, h, a, b, c, d, e, f, 0x2748774C, w2);
    Round(f, g, h, a, b, c, d, e, 0x34B0BCB5, w3);
    Round(e, f, g, h, a, b, c, d, 0x391C0CB3, w4);
    Round(d, e, f, g, h, a, b, c, 0x4ED8AA4A, w5);
    Round(c, d, e, f, g, h, a, b, 0x5B9CCA4F, w6);
    Round(b, c, d, e, f, g, h, a, 0x682E6FF3, w7);
    Round(a, b, c, d, e, f, g, h, 0x748F82EE, w8);
    Round(h, a, b, c, d, e, f, g, 0x78A5636F, w9);
    Round(g, h, a, b, c, d, e, f, 0x84C87814, w10);
    Round(f, g, h, a, b, c, d, e, 0x8CC70208, w11);
    Round(e, f, g, h, a, b, c, d, 0x90BEFFFA, w12);
    Round(d, e, f, g, h, a, b, c, 0xA4506CEB, w13);
    Round(c, d, e, f, g, h, a, b, 0xBEF9A3F7, w14);
    Round(b, c, d, e, f, g, h, a, 0xC67178F2, w15);

    s[0] = _mm_add_epi32(a, s[0]);
    s[1] = _mm_add_epi32(b, s[1]);
    s[2] = _mm_add_epi32(c, s[2]);
    s[3] = _mm_add_epi32(d, s[3]);
    s[4] = _mm_add_epi32(e, s[4]);
    s[5] = _mm_add_epi32(f, s[5]);
    s[6] = _mm_add_epi32(g, s[6]);
    s[7] = _mm_add_epi32(h, s[7]);

  }

} // end namespace

static const uint8_t sizedesc_33[8] = { 0,0,0,0,0,0,1,8 };
static const uint8_t sizedesc_65[8] = { 0,0,0,0,0,0,2,8 };

static const unsigned char pad[64] = { 0x80 };

#define WB(d,i)          \
WRITEBE32(d    ,  s+0,i); \
WRITEBE32(d + 4,  s+1,i); \
WRITEBE32(d + 8,  s+2,i); \
WRITEBE32(d + 12, s+3,i); \
WRITEBE32(d + 16, s+4,i); \
WRITEBE32(d + 20, s+5,i); \
WRITEBE32(d + 24, s+6,i); \
WRITEBE32(d + 28, s+7,i);

void sha256sse_33(
  unsigned char *i0, 
  unsigned char *i1,
  unsigned char *i2,
  unsigned char *i3,
  unsigned char *d0,
  unsigned char *d1,
  unsigned char *d2,
  unsigned char *d3) {

  __m128i s[8];
  uint8_t *bs[] = {i0,i1,i2,i3};

  _sha256sse::Initialize(s);

  memcpy(i0 + 33, pad, 23);
  memcpy(i0 + 56, sizedesc_33, 8);

  memcpy(i1 + 33, pad, 23);
  memcpy(i1 + 56, sizedesc_33, 8);

  memcpy(i2 + 33, pad, 23);
  memcpy(i2 + 56, sizedesc_33, 8);

  memcpy(i3 + 33, pad, 23);
  memcpy(i3 + 56, sizedesc_33, 8);

  _sha256sse::Transform(s, bs);

  WB(d0, 3);
  WB(d1, 2);
  WB(d2, 1);
  WB(d3, 0);

}


void sha256sse_65(
  unsigned char *i0,
  unsigned char *i1,
  unsigned char *i2,
  unsigned char *i3,
  unsigned char *d0,
  unsigned char *d1,
  unsigned char *d2,
  unsigned char *d3) {

  __m128i s[8];
  uint8_t *bs[] = { i0,i1,i2,i3 };
  uint8_t *bs2[] = { i0 + 64,i1 + 64,i2 + 64,i3 + 64 };

  _sha256sse::Initialize(s);

  memcpy(i0 + 65, pad, 55);
  memcpy(i0 + 120, sizedesc_65, 8);

  memcpy(i1 + 65, pad, 55);
  memcpy(i1 + 120, sizedesc_65, 8);

  memcpy(i2 + 65, pad, 55);
  memcpy(i2 + 120, sizedesc_65, 8);

  memcpy(i3 + 65, pad, 55);
  memcpy(i3 + 120, sizedesc_65, 8);

  _sha256sse::Transform(s, bs);
  _sha256sse::Transform(s, bs2);

  WB(d0, 3);
  WB(d1, 2);
  WB(d2, 1);
  WB(d3, 0);

}

void sha256sse_test() {

  unsigned char h0[32];
  unsigned char h1[32];
  unsigned char h2[32];
  unsigned char h3[32];
  unsigned char ch0[32];
  unsigned char ch1[32];
  unsigned char ch2[32];
  unsigned char ch3[32];
  unsigned char m0[64];
  unsigned char m1[64];
  unsigned char m2[64];
  unsigned char m3[64];

  strcpy((char *)m0, "This is a test message to test 01");
  strcpy((char *)m1, "This is a test message to test 02");
  strcpy((char *)m2, "This is a test message to test 03");
  strcpy((char *)m3, "This is a test message to test 04");

  sha256_33(m0, ch0);
  sha256_33(m1, ch1);
  sha256_33(m2, ch2);
  sha256_33(m3, ch3);

  sha256sse_33(m0,m1,m2,m3,h0,h1,h2,h3);

  if((sha256_hex(h0) != sha256_hex(ch0)) ||
     (sha256_hex(h1) != sha256_hex(ch1)) ||
     (sha256_hex(h2) != sha256_hex(ch2)) ||
     (sha256_hex(h3) != sha256_hex(ch3))) {

    printf("SHA() Results Wrong !\n");
    printf("SHA: %s\n", sha256_hex(ch0).c_str());
    printf("SHA: %s\n", sha256_hex(ch1).c_str());
    printf("SHA: %s\n", sha256_hex(ch2).c_str());
    printf("SHA: %s\n\n", sha256_hex(ch3).c_str());
    printf("SSE: %s\n", sha256_hex(h0).c_str());
    printf("SSE: %s\n", sha256_hex(h1).c_str());
    printf("SSE: %s\n", sha256_hex(h2).c_str());
    printf("SSE: %s\n\n", sha256_hex(h3).c_str());

  }

  printf("SHA() Results OK !\n");

}
