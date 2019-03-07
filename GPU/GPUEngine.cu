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

#ifndef WIN64
#include <unistd.h>
#include <stdio.h>
#endif

// ---------------------------------------------------------------------------------
// 256(+64) bits integer CUDA libray for SECPK1
// ---------------------------------------------------------------------------------

// We need 1 extra block for ModInv
#define NBBLOCK 5
#define BIFULLSIZE 40

#include "GPUEngine.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include "../hash/sha256.h"
#include "../hash/ripemd160.h"
#include "../Timer.h"

// Assembly directives
#define UADDO(c, a, b) asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));
#define UADDC(c, a, b) asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));
#define UADD(c, a, b) asm volatile ("addc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define UADDO1(c, a) asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));
#define UADDC1(c, a) asm volatile ("addc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));
#define UADD1(c, a) asm volatile ("addc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));

#define USUBO(c, a, b) asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));
#define USUBC(c, a, b) asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));
#define USUB(c, a, b) asm volatile ("subc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define USUBO1(c, a) asm volatile ("sub.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));
#define USUBC1(c, a) asm volatile ("subc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));
#define USUB1(c, a) asm volatile ("subc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));

#define UMULLO(lo,a, b) asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
#define UMULHI(hi,a, b) asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
#define MADDO(r,a,b,c) asm volatile ("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
#define MADDC(r,a,b,c) asm volatile ("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));
#define MADD(r,a,b,c) asm volatile ("madc.hi.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));

__device__ __constant__ uint64_t _0[] = { 0ULL,0ULL,0ULL,0ULL,0ULL };
__device__ __constant__ uint64_t _1[] = { 1ULL,0ULL,0ULL,0ULL,0ULL };

// Field constant (SECPK1)
__device__ __constant__ uint64_t _P[] = { 0xFFFFFFFEFFFFFC2F,0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF,0ULL };
__device__ __constant__ uint64_t MM64 = 0xD838091DD2253531; // 64bits lsb negative inverse of P (mod 2^64)
#include "GPUGroup.h"

#define HSIZE (GRP_SIZE / 2 - 1)

// ---------------------------------------------------------------------------------------

#define _IsPositive(x) (((int64_t)(x[4]))>=0LL)
#define _IsNegative(x) (((int64_t)(x[4]))<0LL)
#define _IsEqual(a,b)  ((a[4] == b[4]) && (a[3] == b[3]) && (a[2] == b[2]) && (a[1] == b[1]) && (a[0] == b[0]))
#define _IsZero(a)     ((a[4] | a[3] | a[2] | a[1] | a[0]) == 0ULL)
#define _IsOne(a)      ((a[4] == 0ULL) && (a[3] == 0ULL) && (a[2] == 0ULL) && (a[1] == 0ULL) && (a[0] == 1ULL))

#define IDX threadIdx.x

// ---------------------------------------------------------------------------------------

#define Add2(r, a, b) {\
  UADDO(r[0], a[0], b[0]); \
  UADDC(r[1], a[1], b[1]); \
  UADDC(r[2], a[2], b[2]); \
  UADDC(r[3], a[3], b[3]); \
  UADD(r[4], a[4], b[4]);}

// ---------------------------------------------------------------------------------------

#define Add1(r,a) { \
  UADDO1(r[0], a[0]); \
  UADDC1(r[1], a[1]); \
  UADDC1(r[2], a[2]); \
  UADDC1(r[3], a[3]); \
  UADD1(r[4], a[4]);}

// ---------------------------------------------------------------------------------------

#define AddP(r) { \
  UADDO1(r[0], _P[0]); \
  UADDC1(r[1], _P[1]); \
  UADDC1(r[2], _P[2]); \
  UADD1(r[3], _P[3]); \
  r[4]= 0ULL;}

// ---------------------------------------------------------------------------------------

#define Sub2(r,a,b)  {\
  USUBO(r[0], a[0], b[0]); \
  USUBC(r[1], a[1], b[1]); \
  USUBC(r[2], a[2], b[2]); \
  USUBC(r[3], a[3], b[3]); \
  USUB(r[4], a[4], b[4]);}

// ---------------------------------------------------------------------------------------

#define Sub1(r,a) {\
  USUBO1(r[0], a[0]); \
  USUBC1(r[1], a[1]); \
  USUBC1(r[2], a[2]); \
  USUBC1(r[3], a[3]); \
  USUB1(r[4], a[4]);}

// ---------------------------------------------------------------------------------------

#define Sub256(r,a,b)  {\
  USUBO(r[0], a[0], b[0]); \
  USUBC(r[1], a[1], b[1]); \
  USUBC(r[2], a[2], b[2]); \
  USUBC(r[3], a[3], b[3]); \
  USUB(r[4], 0ULL, 0ULL);}

// ---------------------------------------------------------------------------------------

#define Sub2561(r,a) {\
  USUBO1(r[0], a[0]); \
  USUBC1(r[1], a[1]); \
  USUBC1(r[2], a[2]); \
  USUBC1(r[3], a[3]); \
  USUB1(r[4], 0ULL);}

// ---------------------------------------------------------------------------------------

#define Neg(r) Sub2(r, _0, r)

// ---------------------------------------------------------------------------------------

#define Mult2(r, a, b) {\
  UMULLO(r[0],a[0],b); \
  UMULLO(r[1],a[1],b); \
  MADDO(r[1], a[0],b,r[1]); \
  UMULLO(r[2],a[2], b); \
  MADDC(r[2], a[1], b, r[2]); \
  UMULLO(r[3],a[3], b); \
  MADDC(r[3], a[2], b, r[3]); \
  UMULLO(r[4],a[4], b); \
  MADD(r[4], a[3], b, r[4]);}

// ---------------------------------------------------------------------------------------

#define UMult(r, a, b) {\
  UMULLO(r[0],a[0],b); \
  UMULLO(r[1],a[1],b); \
  MADDO(r[1], a[0],b,r[1]); \
  UMULLO(r[2],a[2], b); \
  MADDC(r[2], a[1], b, r[2]); \
  UMULLO(r[3],a[3], b); \
  MADDC(r[3], a[2], b, r[3]); \
  MADD(r[4], a[3], b, 0ULL);}

// ---------------------------------------------------------------------------------------

#define AddC(r,a,carry) {\
  UADDO1(r[0], a[0]); \
  UADDC1(r[1], a[1]); \
  UADDC1(r[2], a[2]); \
  UADDC1(r[3], a[3]); \
  UADDC1(r[4], a[4]); \
  UADDC(carry, 0ULL, 0ULL);}

// ---------------------------------------------------------------------------------------

#define AddAndShift(r, a, b, cH) {\
  UADDO(r[0], a[0], b[0]); \
  UADDC(r[0], a[1], b[1]); \
  UADDC(r[1], a[2], b[2]); \
  UADDC(r[2], a[3], b[3]); \
  UADDC(r[3], a[4], b[4]); \
  UADD(r[4], 0ULL, cH);}

// ---------------------------------------------------------------------------------------

#define Shift64(r, a,cH) {\
  r[0] = a[1]; \
  r[1] = a[2]; \
  r[2] = a[3]; \
  r[3] = a[4]; \
  r[4] = cH;}

// ---------------------------------------------------------------------------------------

#define Load(r, a) {\
  (r)[0] = (a)[0]; \
  (r)[1] = (a)[1]; \
  (r)[2] = (a)[2]; \
  (r)[3] = (a)[3]; \
  (r)[4] = (a)[4];}

// ---------------------------------------------------------------------------------------

#define Load256(r, a) {\
  (r)[0] = (a)[0]; \
  (r)[1] = (a)[1]; \
  (r)[2] = (a)[2]; \
  (r)[3] = (a)[3];}

// ---------------------------------------------------------------------------------------

#define Load256A(r, a) {\
  (r)[0] = (a)[IDX]; \
  (r)[1] = (a)[IDX+NB_TRHEAD_PER_GROUP]; \
  (r)[2] = (a)[IDX+2*NB_TRHEAD_PER_GROUP]; \
  (r)[3] = (a)[IDX+3*NB_TRHEAD_PER_GROUP];}

// ---------------------------------------------------------------------------------------

#define Store256A(r, a) {\
  (r)[IDX] = (a)[0]; \
  (r)[IDX+NB_TRHEAD_PER_GROUP] = (a)[1]; \
  (r)[IDX+2*NB_TRHEAD_PER_GROUP] = (a)[2]; \
  (r)[IDX+3*NB_TRHEAD_PER_GROUP] = (a)[3];}

// ---------------------------------------------------------------------------------------

__device__ void ShiftR(uint64_t *r, uint32_t n) {

  uint64_t b;
  uint64_t rem;
  for (uint32_t i = 0; i < NBBLOCK - 1; i++) {
    rem = r[i + 1] << (64 - n);
    b = (r[i] >> n) | rem;
    r[i] = b;
  }
  // With sign extent
  r[NBBLOCK - 1] = (int64_t)(r[NBBLOCK - 1]) >> n;

}

// ---------------------------------------------------------------------------------------

__device__ void IMult(uint64_t *r, uint64_t *a, int64_t b) {

  uint64_t t[NBBLOCK];

  // Make a positive
  if (b < 0) {
    b = -b;
    Sub2(t, _0, a);
  } else {
    Load(t, a);
  }

  Mult2(r, t, b)

}
// ---------------------------------------------------------------------------------------

__device__ void ModNeg256(uint64_t *r, uint64_t *a) {

  uint64_t t[5];
  USUBO(t[0], 0ULL, a[0]);
  USUBC(t[1], 0ULL, a[1]);
  USUBC(t[2], 0ULL, a[2]);
  USUBC(t[3], 0ULL, a[3]);
  UADDO(r[0], t[0], _P[0]);
  UADDC(r[1], t[1], _P[1]);
  UADDC(r[2], t[2], _P[2]);
  UADD(r[3], t[3], _P[3]);

}

// ---------------------------------------------------------------------------------------

__device__ void ModSub256(uint64_t *r, uint64_t *a, uint64_t *b) {

  uint64_t t;
  USUBO(r[0], a[0], b[0]);
  USUBC(r[1], a[1], b[1]);
  USUBC(r[2], a[2], b[2]);
  USUBC(r[3], a[3], b[3]);
  USUB(t, 0ULL, 0ULL);
  if ((int64_t)t < 0) {
    UADDO1(r[0], _P[0]);
    UADDC1(r[1], _P[1]);
    UADDC1(r[2], _P[2]);
    UADD1(r[3], _P[3]);
  }

}

// ---------------------------------------------------------------------------------------

__device__ void ModAdd256(uint64_t *r, uint64_t *b) {

  uint64_t t[5];
  uint64_t c;
  UADDO(t[0], r[0], b[0]);
  UADDC(t[1], r[1], b[1]);
  UADDC(t[2], r[2], b[2]);
  UADDC(t[3], r[3], b[3]);
  UADD(t[4], 0ULL, 0ULL);
  USUBO(r[0], t[0], _P[0]);
  USUBC(r[1], t[1], _P[1]);
  USUBC(r[2], t[2], _P[2]);
  USUBC(r[3], t[3], _P[3]);
  USUB(c, t[4], 0ULL);
  if ((int64_t)c<0) {
    Load256(r,t);
  }

}

// ---------------------------------------------------------------------------------------

__device__ void ModSub256(uint64_t *r, uint64_t *b) {

  uint64_t t;
  USUBO(r[0], r[0], b[0]);
  USUBC(r[1], r[1], b[1]);
  USUBC(r[2], r[2], b[2]);
  USUBC(r[3], r[3], b[3]);
  USUB(t, 0ULL, 0ULL);
  if ((int64_t)t < 0) {
    UADDO1(r[0], _P[0]);
    UADDC1(r[1], _P[1]);
    UADDC1(r[2], _P[2]);
    UADD1(r[3], _P[3]);
  }

}

// ---------------------------------------------------------------------------------------
#define SWAP_ADD(x,y) x+=y;y-=x;
#define SWAP_SUB(x,y) x-=y;y+=x;
#define IS_EVEN(x) ((x&1LL)==0)
#define MSK62 0x3FFFFFFFFFFFFFFF

__device__ void _ModInv(uint64_t *R) {

  // Compute modular inverse of R mop _P (using 320bits signed integer)
  // 0 < this < P  , P must be odd
  // Return 0 if no inverse

  int64_t  bitCount;
  int64_t  uu, uv, vu, vv;
  int64_t  v0, u0;
  uint64_t r0, s0;
  int64_t  nb0;

  uint64_t u[NBBLOCK];
  uint64_t v[NBBLOCK];
  uint64_t r[NBBLOCK];
  uint64_t s[NBBLOCK];
  uint64_t t1[NBBLOCK];
  uint64_t t2[NBBLOCK];
  uint64_t t3[NBBLOCK];
  uint64_t t4[NBBLOCK];

  Load(u, _P);
  Load(v, R);
  Load(r, _0);
  Load(s, _1);

  // Delayed right shift 62bits

  while (!_IsZero(u)) {

    // u' = (uu*u + uv*v) >> bitCount
    // v' = (vu*u + vv*v) >> bitCount
    // Do not maintain a matrix for r and s, the number of 
    // 'added P' can be easily calculated
    uu = 1; uv = 0;
    vu = 0; vv = 1;

    bitCount = 0LL;
    u0 = (int64_t)u[0];
    v0 = (int64_t)v[0];

    // Slightly optimized Binary XCD loop on native signed integers
    // Stop at 62 bits to avoid uv matrix overfow and loss of sign bit
    while (true) {

      while (IS_EVEN(u0) && (bitCount < 62)) {

        bitCount++;
        u0 >>= 1;
        vu <<= 1;
        vv <<= 1;

      }

      if (bitCount == 62)
        break;

      nb0 = (v0 + u0) & 0x3;
      if (nb0 == 0) {
        SWAP_ADD(uv, vv);
        SWAP_ADD(uu, vu);
        SWAP_ADD(u0, v0);
      } else {
        SWAP_SUB(uv, vv);
        SWAP_SUB(uu, vu);
        SWAP_SUB(u0, v0);
      }

    }

    // Now update BigInt variables

    IMult(t1, u, uu);
    IMult(t2, v, uv);
    IMult(t3, u, vu);
    IMult(t4, v, vv);

    // u = (uu*u + uv*v)
    Add2(u, t1, t2);
    // v = (vu*u + vv*v)
    Add2(v, t3, t4);

    IMult(t1, r, uu);
    IMult(t2, s, uv);
    IMult(t3, r, vu);
    IMult(t4, s, vv);

    // Compute multiple of P to add to s and r to make them multiple of 2^62
    r0 = ((t1[0] + t2[0]) * MM64) & MSK62;
    s0 = ((t3[0] + t4[0]) * MM64) & MSK62;
    // r = (uu*r + uv*s + r0*P)
    UMult(r, _P, r0);
    Add1(r, t1);
    Add1(r, t2);

    // s = (vu*r + vv*s + s0*P)
    UMult(s, _P, s0);
    Add1(s, t3);
    Add1(s, t4);

    // Right shift all variables by 62bits
    ShiftR(u, 62);
    ShiftR(v, 62);
    ShiftR(r, 62);
    ShiftR(s, 62);

  }

  // v ends with -1 or 1
  if (_IsNegative(v)) {
    // V = -1
    Sub2(s, _P, s);
    Neg(v);
  }

  if (!_IsOne(v)) {
    // No inverse
    Load(R, _0);
    return;
  }

  if (_IsNegative(s)) {
    AddP(s);
  } else {
    Sub1(s, _P);
    if (_IsNegative(s)) 
      AddP(s);
  }

  Load(R, s);

}

// ---------------------------------------------------------------------------------------
// Compute a*b*(mod n)
// a and b must be lower than n
// ---------------------------------------------------------------------------------------

__device__ void _ModMult(uint64_t *r, uint64_t *a, uint64_t *b) {

  uint64_t r512[8];
  uint64_t t[NBBLOCK];
  uint64_t ah,al;
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;

  // 256*256 multiplier
  UMult(r512, a, b[0]);
  UMult(t, a, b[1]);
  UADDO1(r512[1], t[0]);
  UADDC1(r512[2], t[1]);
  UADDC1(r512[3], t[2]);
  UADDC1(r512[4], t[3]);
  UADD1(r512[5], t[4]);
  UMult(t, a, b[2]);
  UADDO1(r512[2], t[0]);
  UADDC1(r512[3], t[1]);
  UADDC1(r512[4], t[2]);
  UADDC1(r512[5], t[3]);
  UADD1(r512[6], t[4]);
  UMult(t, a, b[3]);
  UADDO1(r512[3], t[0]);
  UADDC1(r512[4], t[1]);
  UADDC1(r512[5], t[2]);
  UADDC1(r512[6], t[3]);
  UADD1(r512[7], t[4]);
 
  // Reduce from 512 to 320 
  UMult(t,(r512 + 4), 0x1000003D1ULL);
  UADDO1(r512[0], t[0]);
  UADDC1(r512[1], t[1]);
  UADDC1(r512[2], t[2]);
  UADDC1(r512[3], t[3]);

  // Reduce from 320 to 256 
  UADD1(t[4],0ULL);
  UMULLO(al,t[4], 0x1000003D1ULL);
  UMULHI(ah,t[4], 0x1000003D1ULL);
  UADDO(r[0],r512[0], al);
  UADDC(r[1],r512[1], ah);
  UADDC(r[2],r512[2], 0ULL);
  UADDC(r[3],r512[3], 0ULL);
  UADD(r[4],0ULL, 0ULL);

}


__device__ void _ModMult(uint64_t *r, uint64_t *a) {

  uint64_t r512[8];
  uint64_t t[NBBLOCK];
  uint64_t ah, al;
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;

  // 256*256 multiplier
  UMult(r512, a, r[0]);
  UMult(t, a, r[1]);
  UADDO1(r512[1], t[0]);
  UADDC1(r512[2], t[1]);
  UADDC1(r512[3], t[2]);
  UADDC1(r512[4], t[3]);
  UADD1(r512[5], t[4]);
  UMult(t, a, r[2]);
  UADDO1(r512[2], t[0]);
  UADDC1(r512[3], t[1]);
  UADDC1(r512[4], t[2]);
  UADDC1(r512[5], t[3]);
  UADD1(r512[6], t[4]);
  UMult(t, a, r[3]);
  UADDO1(r512[3], t[0]);
  UADDC1(r512[4], t[1]);
  UADDC1(r512[5], t[2]);
  UADDC1(r512[6], t[3]);
  UADD1(r512[7], t[4]);

  // Reduce from 512 to 320 
  UMult(t, (r512 + 4), 0x1000003D1ULL);
  UADDO1(r512[0], t[0]);
  UADDC1(r512[1], t[1]);
  UADDC1(r512[2], t[2]);
  UADDC1(r512[3], t[3]);

  // Reduce from 320 to 256
  UADD1(t[4], 0ULL);
  UMULLO(al, t[4], 0x1000003D1ULL);
  UMULHI(ah, t[4], 0x1000003D1ULL);
  UADDO(r[0],r512[0], al);
  UADDC(r[1],r512[1], ah);
  UADDC(r[2],r512[2], 0ULL);
  UADDC(r[3],r512[3], 0ULL);
  UADD(r[4], 0ULL, 0ULL);

}


// ---------------------------------------------------------------------------------------
// Compute all ModInv of the group
// ---------------------------------------------------------------------------------------

__device__ void _ModInvGrouped(uint64_t r[GRP_SIZE / 2 + 1][4]) {

  uint64_t subp[GRP_SIZE / 2 + 1][4];
  uint64_t newValue[4];
  uint64_t inverse[5];

  Load256(subp[0], r[0]);
  for (uint32_t i = 1; i < (GRP_SIZE / 2 + 1); i++) {
    _ModMult(subp[i], subp[i - 1], r[i]);
  }

  // We need 320bit signed int for ModInv
  Load256(inverse, subp[(GRP_SIZE / 2 + 1) - 1]);
  inverse[4] = 0;
  _ModInv(inverse);

  for (uint32_t i = (GRP_SIZE / 2 + 1) - 1; i > 0; i--) {
    _ModMult(newValue, subp[i - 1], inverse);
    _ModMult(inverse, r[i]);
    Load256(r[i], newValue);
  }

  Load256(r[0], inverse);

}

// ---------------------------------------------------------------------------------
// SHA256
// ---------------------------------------------------------------------------------

__device__ __constant__ uint32_t K[] =
{
    0x428A2F98, 0x71374491, 0xB5C0FBCF, 0xE9B5DBA5,
    0x3956C25B, 0x59F111F1, 0x923F82A4, 0xAB1C5ED5,
    0xD807AA98, 0x12835B01, 0x243185BE, 0x550C7DC3,
    0x72BE5D74, 0x80DEB1FE, 0x9BDC06A7, 0xC19BF174,
    0xE49B69C1, 0xEFBE4786, 0x0FC19DC6, 0x240CA1CC,
    0x2DE92C6F, 0x4A7484AA, 0x5CB0A9DC, 0x76F988DA,
    0x983E5152, 0xA831C66D, 0xB00327C8, 0xBF597FC7,
    0xC6E00BF3, 0xD5A79147, 0x06CA6351, 0x14292967,
    0x27B70A85, 0x2E1B2138, 0x4D2C6DFC, 0x53380D13,
    0x650A7354, 0x766A0ABB, 0x81C2C92E, 0x92722C85,
    0xA2BFE8A1, 0xA81A664B, 0xC24B8B70, 0xC76C51A3,
    0xD192E819, 0xD6990624, 0xF40E3585, 0x106AA070,
    0x19A4C116, 0x1E376C08, 0x2748774C, 0x34B0BCB5,
    0x391C0CB3, 0x4ED8AA4A, 0x5B9CCA4F, 0x682E6FF3,
    0x748F82EE, 0x78A5636F, 0x84C87814, 0x8CC70208,
    0x90BEFFFA, 0xA4506CEB, 0xBEF9A3F7, 0xC67178F2,
};

__device__ __constant__ uint32_t I[] = {
  0x6a09e667ul,
  0xbb67ae85ul,
  0x3c6ef372ul,
  0xa54ff53aul,
  0x510e527ful,
  0x9b05688cul,
  0x1f83d9abul,
  0x5be0cd19ul,
};

//#define ASSEMBLY_SIGMA
#ifdef ASSEMBLY_SIGMA

__device__ __forceinline__ uint32_t S0(uint32_t x) {

  uint32_t y;
  asm("{\n\t" 
      " .reg .u64 r1,r2,r3;\n\t"
      " cvt.u64.u32 r1, %1;\n\t"
      " mov.u64 r2, r1;\n\t"
      " shl.b64 r2, r2,32;\n\t"
      " or.b64  r1, r1,r2;\n\t"
      " shr.b64 r3, r1, 2;\n\t"
      " mov.u64 r2, r3;\n\t"
      " shr.b64 r3, r1, 13;\n\t"
      " xor.b64 r2, r2, r3;\n\t"
      " shr.b64 r3, r1, 22;\n\t"
      " xor.b64 r2, r2, r3;\n\t"
      " cvt.u32.u64 %0,r2;\n\t"
      "}\n\t"
    : "=r"(y) : "r" (x));
  return y;

}

__device__ __forceinline__ uint32_t S1(uint32_t x) {

  uint32_t y;
  asm("{\n\t"
    " .reg .u64 r1,r2,r3;\n\t"
    " cvt.u64.u32 r1, %1;\n\t"
    " mov.u64 r2, r1;\n\t"
    " shl.b64 r2, r2,32;\n\t"
    " or.b64  r1, r1,r2;\n\t"
    " shr.b64 r3, r1, 6;\n\t"
    " mov.u64 r2, r3;\n\t"
    " shr.b64 r3, r1, 11;\n\t"
    " xor.b64 r2, r2, r3;\n\t"
    " shr.b64 r3, r1, 25;\n\t"
    " xor.b64 r2, r2, r3;\n\t"
    " cvt.u32.u64 %0,r2;\n\t"
    "}\n\t"
    : "=r"(y) : "r" (x));
  return y;

}

__device__ __forceinline__ uint32_t s0(uint32_t x) {

  uint32_t y;
  asm("{\n\t"
    " .reg .u64 r1,r2,r3;\n\t"
    " cvt.u64.u32 r1, %1;\n\t"
    " mov.u64 r2, r1;\n\t"
    " shl.b64 r2, r2,32;\n\t"
    " or.b64  r1, r1,r2;\n\t"
    " shr.b64 r2, r2, 35;\n\t"
    " shr.b64 r3, r1, 18;\n\t"
    " xor.b64 r2, r2, r3;\n\t"
    " shr.b64 r3, r1, 7;\n\t"
    " xor.b64 r2, r2, r3;\n\t"
    " cvt.u32.u64 %0,r2;\n\t"
    "}\n\t"
    : "=r"(y) : "r" (x));
  return y;

}

__device__ __forceinline__ uint32_t s1(uint32_t x) {

  uint32_t y;
  asm("{\n\t"
    " .reg .u64 r1,r2,r3;\n\t"
    " cvt.u64.u32 r1, %1;\n\t"
    " mov.u64 r2, r1;\n\t"
    " shl.b64 r2, r2,32;\n\t"
    " or.b64  r1, r1,r2;\n\t"
    " shr.b64 r2, r2, 42;\n\t"
    " shr.b64 r3, r1, 19;\n\t"
    " xor.b64 r2, r2, r3;\n\t"
    " shr.b64 r3, r1, 17;\n\t"
    " xor.b64 r2, r2, r3;\n\t"
    " cvt.u32.u64 %0,r2;\n\t"
    "}\n\t"
    : "=r"(y) : "r" (x));
  return y;

}

#else

#define ROR(x,n) ((x>>n)|(x<<(32-n)))
#define S0(x) (ROR(x,2) ^ ROR(x,13) ^ ROR(x,22))
#define S1(x) (ROR(x,6) ^ ROR(x,11) ^ ROR(x,25))
#define s0(x) (ROR(x,7) ^ ROR(x,18) ^ (x >> 3))
#define s1(x) (ROR(x,17) ^ ROR(x,19) ^ (x >> 10))

#endif

//#define Maj(x,y,z) ((x&y)^(x&z)^(y&z))
//#define Ch(x,y,z)  ((x&y)^(~x&z))

// The following functions are equivalent to the above
#define Maj(x,y,z) ((x & y) | (z & (x | y)))
#define Ch(x,y,z) (z ^ (x & (y ^ z)))

// SHA-256 inner round
#define S2Round(a, b, c, d, e, f, g, h, k, w) \
    t1 = h + S1(e) + Ch(e,f,g) + k + (w); \
    t2 = S0(a) + Maj(a,b,c); \
    d += t1; \
    h = t1 + t2;

// WMIX
#define WMIX() { \
w[0] += s1(w[14]) + w[9] + s0(w[1]);\
w[1] += s1(w[15]) + w[10] + s0(w[2]);\
w[2] += s1(w[0]) + w[11] + s0(w[3]);\
w[3] += s1(w[1]) + w[12] + s0(w[4]);\
w[4] += s1(w[2]) + w[13] + s0(w[5]);\
w[5] += s1(w[3]) + w[14] + s0(w[6]);\
w[6] += s1(w[4]) + w[15] + s0(w[7]);\
w[7] += s1(w[5]) + w[0] + s0(w[8]);\
w[8] += s1(w[6]) + w[1] + s0(w[9]);\
w[9] += s1(w[7]) + w[2] + s0(w[10]);\
w[10] += s1(w[8]) + w[3] + s0(w[11]);\
w[11] += s1(w[9]) + w[4] + s0(w[12]);\
w[12] += s1(w[10]) + w[5] + s0(w[13]);\
w[13] += s1(w[11]) + w[6] + s0(w[14]);\
w[14] += s1(w[12]) + w[7] + s0(w[15]);\
w[15] += s1(w[13]) + w[8] + s0(w[0]);\
}

// ROUND
#define SHA256_RND(k) {\
S2Round(a, b, c, d, e, f, g, h, K[k], w[0]);\
S2Round(h, a, b, c, d, e, f, g, K[k + 1], w[1]);\
S2Round(g, h, a, b, c, d, e, f, K[k + 2], w[2]);\
S2Round(f, g, h, a, b, c, d, e, K[k + 3], w[3]);\
S2Round(e, f, g, h, a, b, c, d, K[k + 4], w[4]);\
S2Round(d, e, f, g, h, a, b, c, K[k + 5], w[5]);\
S2Round(c, d, e, f, g, h, a, b, K[k + 6], w[6]);\
S2Round(b, c, d, e, f, g, h, a, K[k + 7], w[7]);\
S2Round(a, b, c, d, e, f, g, h, K[k + 8], w[8]);\
S2Round(h, a, b, c, d, e, f, g, K[k + 9], w[9]);\
S2Round(g, h, a, b, c, d, e, f, K[k + 10], w[10]);\
S2Round(f, g, h, a, b, c, d, e, K[k + 11], w[11]);\
S2Round(e, f, g, h, a, b, c, d, K[k + 12], w[12]);\
S2Round(d, e, f, g, h, a, b, c, K[k + 13], w[13]);\
S2Round(c, d, e, f, g, h, a, b, K[k + 14], w[14]);\
S2Round(b, c, d, e, f, g, h, a, K[k + 15], w[15]);\
}

//#define bswap32(v) (((v) >> 24) | (((v) >> 8) & 0xff00) | (((v) << 8) & 0xff0000) | ((v) << 24))
#define bswap32(v) __byte_perm(v, 0, 0x0123)

// Initialise state
__device__ void SHA256Initialize(uint32_t s[8]) {
#pragma unroll 8
  for (int i = 0; i < 8; i++)
    s[i] = I[i];
}

#define DEF(x,y) uint32_t x = s[y]

// Perform SHA-256 transformations, process 64-byte chunks
__device__ void SHA256Transform(uint32_t s[8],uint32_t* w) {
  
  uint32_t t1;
  uint32_t t2;

  DEF(a, 0);
  DEF(b, 1);
  DEF(c, 2);
  DEF(d, 3);
  DEF(e, 4);
  DEF(f, 5);
  DEF(g, 6);
  DEF(h, 7);

  SHA256_RND(0);
  WMIX();
  SHA256_RND(16);
  WMIX();
  SHA256_RND(32);
  WMIX();
  SHA256_RND(48);

  s[0] += a;
  s[1] += b;
  s[2] += c;
  s[3] += d;
  s[4] += e;
  s[5] += f;
  s[6] += g;
  s[7] += h;

}


// ---------------------------------------------------------------------------------
// RIPEMD160
// ---------------------------------------------------------------------------------
__device__ __constant__ uint64_t ripemd160_sizedesc_32 = 32 << 3;

__device__ void RIPEMD160Initialize(uint32_t s[5]) {

  s[0] = 0x67452301ul;
  s[1] = 0xEFCDAB89ul;
  s[2] = 0x98BADCFEul;
  s[3] = 0x10325476ul;
  s[4] = 0xC3D2E1F0ul;

}

#define ROL(x,n) ((x>>(32-n))|(x<<n))
#define f1(x, y, z) (x ^ y ^ z)
#define f2(x, y, z) ((x & y) | (~x & z))
#define f3(x, y, z) ((x | ~y) ^ z)
#define f4(x, y, z) ((x & z) | (~z & y))
#define f5(x, y, z) (x ^ (y | ~z))

#define RPRound(a,b,c,d,e,f,x,k,r) \
  u = a + f + x + k; \
  a = ROL(u, r) + e; \
  c = ROL(c, 10);

#define R11(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f1(b, c, d), x, 0, r)
#define R21(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f2(b, c, d), x, 0x5A827999ul, r)
#define R31(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f3(b, c, d), x, 0x6ED9EBA1ul, r)
#define R41(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f4(b, c, d), x, 0x8F1BBCDCul, r)
#define R51(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f5(b, c, d), x, 0xA953FD4Eul, r)
#define R12(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f5(b, c, d), x, 0x50A28BE6ul, r)
#define R22(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f4(b, c, d), x, 0x5C4DD124ul, r)
#define R32(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f3(b, c, d), x, 0x6D703EF3ul, r)
#define R42(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f2(b, c, d), x, 0x7A6D76E9ul, r)
#define R52(a,b,c,d,e,x,r) RPRound(a, b, c, d, e, f1(b, c, d), x, 0, r)

/** Perform a RIPEMD-160 transformation, processing a 64-byte chunk. */
__device__ void RIPEMD160Transform(uint32_t s[5],uint32_t* w) {

  uint32_t u;
  uint32_t a1 = s[0], b1 = s[1], c1 = s[2], d1 = s[3], e1 = s[4];
  uint32_t a2 = a1, b2 = b1, c2 = c1, d2 = d1, e2 = e1;

  R11(a1, b1, c1, d1, e1, w[0], 11);
  R12(a2, b2, c2, d2, e2, w[5], 8);
  R11(e1, a1, b1, c1, d1, w[1], 14);
  R12(e2, a2, b2, c2, d2, w[14], 9);
  R11(d1, e1, a1, b1, c1, w[2], 15);
  R12(d2, e2, a2, b2, c2, w[7], 9);
  R11(c1, d1, e1, a1, b1, w[3], 12);
  R12(c2, d2, e2, a2, b2, w[0], 11);
  R11(b1, c1, d1, e1, a1, w[4], 5);
  R12(b2, c2, d2, e2, a2, w[9], 13);
  R11(a1, b1, c1, d1, e1, w[5], 8);
  R12(a2, b2, c2, d2, e2, w[2], 15);
  R11(e1, a1, b1, c1, d1, w[6], 7);
  R12(e2, a2, b2, c2, d2, w[11], 15);
  R11(d1, e1, a1, b1, c1, w[7], 9);
  R12(d2, e2, a2, b2, c2, w[4], 5);
  R11(c1, d1, e1, a1, b1, w[8], 11);
  R12(c2, d2, e2, a2, b2, w[13], 7);
  R11(b1, c1, d1, e1, a1, w[9], 13);
  R12(b2, c2, d2, e2, a2, w[6], 7);
  R11(a1, b1, c1, d1, e1, w[10], 14);
  R12(a2, b2, c2, d2, e2, w[15], 8);
  R11(e1, a1, b1, c1, d1, w[11], 15);
  R12(e2, a2, b2, c2, d2, w[8], 11);
  R11(d1, e1, a1, b1, c1, w[12], 6);
  R12(d2, e2, a2, b2, c2, w[1], 14);
  R11(c1, d1, e1, a1, b1, w[13], 7);
  R12(c2, d2, e2, a2, b2, w[10], 14);
  R11(b1, c1, d1, e1, a1, w[14], 9);
  R12(b2, c2, d2, e2, a2, w[3], 12);
  R11(a1, b1, c1, d1, e1, w[15], 8);
  R12(a2, b2, c2, d2, e2, w[12], 6);

  R21(e1, a1, b1, c1, d1, w[7], 7);
  R22(e2, a2, b2, c2, d2, w[6], 9);
  R21(d1, e1, a1, b1, c1, w[4], 6);
  R22(d2, e2, a2, b2, c2, w[11], 13);
  R21(c1, d1, e1, a1, b1, w[13], 8);
  R22(c2, d2, e2, a2, b2, w[3], 15);
  R21(b1, c1, d1, e1, a1, w[1], 13);
  R22(b2, c2, d2, e2, a2, w[7], 7);
  R21(a1, b1, c1, d1, e1, w[10], 11);
  R22(a2, b2, c2, d2, e2, w[0], 12);
  R21(e1, a1, b1, c1, d1, w[6], 9);
  R22(e2, a2, b2, c2, d2, w[13], 8);
  R21(d1, e1, a1, b1, c1, w[15], 7);
  R22(d2, e2, a2, b2, c2, w[5], 9);
  R21(c1, d1, e1, a1, b1, w[3], 15);
  R22(c2, d2, e2, a2, b2, w[10], 11);
  R21(b1, c1, d1, e1, a1, w[12], 7);
  R22(b2, c2, d2, e2, a2, w[14], 7);
  R21(a1, b1, c1, d1, e1, w[0], 12);
  R22(a2, b2, c2, d2, e2, w[15], 7);
  R21(e1, a1, b1, c1, d1, w[9], 15);
  R22(e2, a2, b2, c2, d2, w[8], 12);
  R21(d1, e1, a1, b1, c1, w[5], 9);
  R22(d2, e2, a2, b2, c2, w[12], 7);
  R21(c1, d1, e1, a1, b1, w[2], 11);
  R22(c2, d2, e2, a2, b2, w[4], 6);
  R21(b1, c1, d1, e1, a1, w[14], 7);
  R22(b2, c2, d2, e2, a2, w[9], 15);
  R21(a1, b1, c1, d1, e1, w[11], 13);
  R22(a2, b2, c2, d2, e2, w[1], 13);
  R21(e1, a1, b1, c1, d1, w[8], 12);
  R22(e2, a2, b2, c2, d2, w[2], 11);

  R31(d1, e1, a1, b1, c1, w[3], 11);
  R32(d2, e2, a2, b2, c2, w[15], 9);
  R31(c1, d1, e1, a1, b1, w[10], 13);
  R32(c2, d2, e2, a2, b2, w[5], 7);
  R31(b1, c1, d1, e1, a1, w[14], 6);
  R32(b2, c2, d2, e2, a2, w[1], 15);
  R31(a1, b1, c1, d1, e1, w[4], 7);
  R32(a2, b2, c2, d2, e2, w[3], 11);
  R31(e1, a1, b1, c1, d1, w[9], 14);
  R32(e2, a2, b2, c2, d2, w[7], 8);
  R31(d1, e1, a1, b1, c1, w[15], 9);
  R32(d2, e2, a2, b2, c2, w[14], 6);
  R31(c1, d1, e1, a1, b1, w[8], 13);
  R32(c2, d2, e2, a2, b2, w[6], 6);
  R31(b1, c1, d1, e1, a1, w[1], 15);
  R32(b2, c2, d2, e2, a2, w[9], 14);
  R31(a1, b1, c1, d1, e1, w[2], 14);
  R32(a2, b2, c2, d2, e2, w[11], 12);
  R31(e1, a1, b1, c1, d1, w[7], 8);
  R32(e2, a2, b2, c2, d2, w[8], 13);
  R31(d1, e1, a1, b1, c1, w[0], 13);
  R32(d2, e2, a2, b2, c2, w[12], 5);
  R31(c1, d1, e1, a1, b1, w[6], 6);
  R32(c2, d2, e2, a2, b2, w[2], 14);
  R31(b1, c1, d1, e1, a1, w[13], 5);
  R32(b2, c2, d2, e2, a2, w[10], 13);
  R31(a1, b1, c1, d1, e1, w[11], 12);
  R32(a2, b2, c2, d2, e2, w[0], 13);
  R31(e1, a1, b1, c1, d1, w[5], 7);
  R32(e2, a2, b2, c2, d2, w[4], 7);
  R31(d1, e1, a1, b1, c1, w[12], 5);
  R32(d2, e2, a2, b2, c2, w[13], 5);

  R41(c1, d1, e1, a1, b1, w[1], 11);
  R42(c2, d2, e2, a2, b2, w[8], 15);
  R41(b1, c1, d1, e1, a1, w[9], 12);
  R42(b2, c2, d2, e2, a2, w[6], 5);
  R41(a1, b1, c1, d1, e1, w[11], 14);
  R42(a2, b2, c2, d2, e2, w[4], 8);
  R41(e1, a1, b1, c1, d1, w[10], 15);
  R42(e2, a2, b2, c2, d2, w[1], 11);
  R41(d1, e1, a1, b1, c1, w[0], 14);
  R42(d2, e2, a2, b2, c2, w[3], 14);
  R41(c1, d1, e1, a1, b1, w[8], 15);
  R42(c2, d2, e2, a2, b2, w[11], 14);
  R41(b1, c1, d1, e1, a1, w[12], 9);
  R42(b2, c2, d2, e2, a2, w[15], 6);
  R41(a1, b1, c1, d1, e1, w[4], 8);
  R42(a2, b2, c2, d2, e2, w[0], 14);
  R41(e1, a1, b1, c1, d1, w[13], 9);
  R42(e2, a2, b2, c2, d2, w[5], 6);
  R41(d1, e1, a1, b1, c1, w[3], 14);
  R42(d2, e2, a2, b2, c2, w[12], 9);
  R41(c1, d1, e1, a1, b1, w[7], 5);
  R42(c2, d2, e2, a2, b2, w[2], 12);
  R41(b1, c1, d1, e1, a1, w[15], 6);
  R42(b2, c2, d2, e2, a2, w[13], 9);
  R41(a1, b1, c1, d1, e1, w[14], 8);
  R42(a2, b2, c2, d2, e2, w[9], 12);
  R41(e1, a1, b1, c1, d1, w[5], 6);
  R42(e2, a2, b2, c2, d2, w[7], 5);
  R41(d1, e1, a1, b1, c1, w[6], 5);
  R42(d2, e2, a2, b2, c2, w[10], 15);
  R41(c1, d1, e1, a1, b1, w[2], 12);
  R42(c2, d2, e2, a2, b2, w[14], 8);

  R51(b1, c1, d1, e1, a1, w[4], 9);
  R52(b2, c2, d2, e2, a2, w[12], 8);
  R51(a1, b1, c1, d1, e1, w[0], 15);
  R52(a2, b2, c2, d2, e2, w[15], 5);
  R51(e1, a1, b1, c1, d1, w[5], 5);
  R52(e2, a2, b2, c2, d2, w[10], 12);
  R51(d1, e1, a1, b1, c1, w[9], 11);
  R52(d2, e2, a2, b2, c2, w[4], 9);
  R51(c1, d1, e1, a1, b1, w[7], 6);
  R52(c2, d2, e2, a2, b2, w[1], 12);
  R51(b1, c1, d1, e1, a1, w[12], 8);
  R52(b2, c2, d2, e2, a2, w[5], 5);
  R51(a1, b1, c1, d1, e1, w[2], 13);
  R52(a2, b2, c2, d2, e2, w[8], 14);
  R51(e1, a1, b1, c1, d1, w[10], 12);
  R52(e2, a2, b2, c2, d2, w[7], 6);
  R51(d1, e1, a1, b1, c1, w[14], 5);
  R52(d2, e2, a2, b2, c2, w[6], 8);
  R51(c1, d1, e1, a1, b1, w[1], 12);
  R52(c2, d2, e2, a2, b2, w[2], 13);
  R51(b1, c1, d1, e1, a1, w[3], 13);
  R52(b2, c2, d2, e2, a2, w[13], 6);
  R51(a1, b1, c1, d1, e1, w[8], 14);
  R52(a2, b2, c2, d2, e2, w[14], 5);
  R51(e1, a1, b1, c1, d1, w[11], 11);
  R52(e2, a2, b2, c2, d2, w[0], 15);
  R51(d1, e1, a1, b1, c1, w[6], 8);
  R52(d2, e2, a2, b2, c2, w[3], 13);
  R51(c1, d1, e1, a1, b1, w[15], 5);
  R52(c2, d2, e2, a2, b2, w[9], 11);
  R51(b1, c1, d1, e1, a1, w[13], 6);
  R52(b2, c2, d2, e2, a2, w[11], 11);

  uint32_t t = s[0];
  s[0] = s[1] + c1 + d2;
  s[1] = s[2] + d1 + e2;
  s[2] = s[3] + e1 + a2;
  s[3] = s[4] + a1 + b2;
  s[4] = t + b1 + c2;
}

// ---------------------------------------------------------------------------------
// EC
// ---------------------------------------------------------------------------------

__device__ void _GetHash160Comp(uint64_t *x, uint64_t *y, uint8_t *hash) {

  uint32_t *x32 = (uint32_t *)(x);
  uint32_t publicKeyBytes[16];
  uint32_t s[16];

  // Compressed public key
  publicKeyBytes[0] = __byte_perm(x32[7], 0x2 + (y[0] & 1) , 0x4321 );
  publicKeyBytes[1] = __byte_perm(x32[7], x32[6], 0x0765);
  publicKeyBytes[2] = __byte_perm(x32[6], x32[5], 0x0765);
  publicKeyBytes[3] = __byte_perm(x32[5], x32[4], 0x0765);
  publicKeyBytes[4] = __byte_perm(x32[4], x32[3], 0x0765);
  publicKeyBytes[5] = __byte_perm(x32[3], x32[2], 0x0765);
  publicKeyBytes[6] = __byte_perm(x32[2], x32[1], 0x0765);
  publicKeyBytes[7] = __byte_perm(x32[1], x32[0], 0x0765);
  publicKeyBytes[8] = __byte_perm(x32[0], 0x80, 0x0456);
  publicKeyBytes[9] = 0;
  publicKeyBytes[10] = 0;
  publicKeyBytes[11] = 0;
  publicKeyBytes[12] = 0;
  publicKeyBytes[13] = 0;
  publicKeyBytes[14] = 0;
  publicKeyBytes[15] = 0x108;

  SHA256Initialize(s);
  SHA256Transform(s, publicKeyBytes);

#pragma unroll 8
  for (int i = 0; i < 8; i++)
    s[i] = bswap32(s[i]);

  *(uint64_t *)(s +  8) = 0x80ULL;
  *(uint64_t *)(s + 10) = 0ULL;
  *(uint64_t *)(s + 12) = 0ULL;
  *(uint64_t *)(s + 14) = ripemd160_sizedesc_32;

  RIPEMD160Initialize((uint32_t *)hash);
  RIPEMD160Transform((uint32_t *)hash,s);

}

__device__ void _GetHash160(uint64_t *x, uint64_t *y, uint8_t *hash) {

  uint32_t *x32 = (uint32_t *)(x);
  uint32_t *y32 = (uint32_t *)(y);
  uint32_t publicKeyBytes[32];
  uint32_t s[16];

  // Uncompressed public key
  publicKeyBytes[0] = __byte_perm(x32[7], 0x04, 0x4321);
  publicKeyBytes[1] = __byte_perm(x32[7], x32[6], 0x0765);
  publicKeyBytes[2] = __byte_perm(x32[6], x32[5], 0x0765);
  publicKeyBytes[3] = __byte_perm(x32[5], x32[4], 0x0765);
  publicKeyBytes[4] = __byte_perm(x32[4], x32[3], 0x0765);
  publicKeyBytes[5] = __byte_perm(x32[3], x32[2], 0x0765);
  publicKeyBytes[6] = __byte_perm(x32[2], x32[1], 0x0765);
  publicKeyBytes[7] = __byte_perm(x32[1], x32[0], 0x0765);
  publicKeyBytes[8] = __byte_perm(x32[0], y32[7], 0x0765);
  publicKeyBytes[9] = __byte_perm(y32[7], y32[6], 0x0765);
  publicKeyBytes[10] = __byte_perm(y32[6], y32[5], 0x0765);
  publicKeyBytes[11] = __byte_perm(y32[5], y32[4], 0x0765);
  publicKeyBytes[12] = __byte_perm(y32[4], y32[3], 0x0765);
  publicKeyBytes[13] = __byte_perm(y32[3], y32[2], 0x0765);
  publicKeyBytes[14] = __byte_perm(y32[2], y32[1], 0x0765);
  publicKeyBytes[15] = __byte_perm(y32[1], y32[0], 0x0765);
  publicKeyBytes[16] = __byte_perm(y32[0], 0x80, 0x0456);
  publicKeyBytes[17] = 0;
  publicKeyBytes[18] = 0;
  publicKeyBytes[19] = 0;
  publicKeyBytes[20] = 0;
  publicKeyBytes[21] = 0;
  publicKeyBytes[22] = 0;
  publicKeyBytes[23] = 0;
  publicKeyBytes[24] = 0;
  publicKeyBytes[25] = 0;
  publicKeyBytes[26] = 0;
  publicKeyBytes[27] = 0;
  publicKeyBytes[28] = 0;
  publicKeyBytes[29] = 0;
  publicKeyBytes[30] = 0;
  publicKeyBytes[31] = 0x208;

  SHA256Initialize(s);
  SHA256Transform(s, publicKeyBytes);
  SHA256Transform(s, publicKeyBytes+16);

#pragma unroll 8
  for (int i = 0; i < 8; i++)
    s[i] = bswap32(s[i]);

  *(uint64_t *)(s + 8) = 0x80ULL;
  *(uint64_t *)(s + 10) = 0ULL;
  *(uint64_t *)(s + 12) = 0ULL;
  *(uint64_t *)(s + 14) = ripemd160_sizedesc_32;

  RIPEMD160Initialize((uint32_t *)hash);
  RIPEMD160Transform((uint32_t *)hash, s);

}


// ---------------------------------------------------------------------------------------

#define __COMPFUNC__ ComputeKeysComp
#define __HASHFUNC__ _GetHash160Comp
#include "GPUCompute.h"
#undef __COMPFUNC__
#undef __HASHFUNC__
#define __COMPFUNC__ ComputeKeysUncomp
#define __HASHFUNC__ _GetHash160
#include "GPUCompute.h"


// ---------------------------------------------------------------------------------------

__global__ void comp_keys_comp(uint16_t prefix, uint64_t *keys, uint8_t *found) {

  int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
  int xPtr = (blockIdx.x*blockDim.x)*8;
  int yPtr = xPtr + 4*NB_TRHEAD_PER_GROUP;
  ComputeKeysComp(keys+xPtr, keys+yPtr, prefix, found+tid*MEMOUT_PER_THREAD);

}

// ---------------------------------------------------------------------------------------

__global__ void comp_keys_uncomp(uint16_t prefix, uint64_t *keys, uint8_t *found) {

  int tid = (blockIdx.x*blockDim.x) + threadIdx.x;
  int xPtr = (blockIdx.x*blockDim.x) * 8;
  int yPtr = xPtr + 4 * NB_TRHEAD_PER_GROUP;
  ComputeKeysUncomp(keys + xPtr, keys + yPtr, prefix, found + tid * MEMOUT_PER_THREAD);

}

// ---------------------------------------------------------------------------------------

__global__ void chekc_mult(uint64_t *a, uint64_t *b, uint64_t *r) {

  _ModMult(r, a, b);
  r[4]=0;

}

// ---------------------------------------------------------------------------------------

__global__ void chekc_hash160(uint64_t *x, uint64_t *y, uint8_t *h) {

  _GetHash160Comp(x, y, h);

}

// ---------------------------------------------------------------------------------------

__global__ void get_endianness(uint8_t *endian) {

  uint32_t a = 0x01020304;
  uint8_t fb = *(uint8_t *)(&a);
  *endian = (fb==0x04);

}

// ---------------------------------------------------------------------------------------

using namespace std;

double Comb(int n, int k) {

  double A = 1.0;
  for(int i=n;i>=(n-k+1);i--)
    A = A * (double)i;

  double K = 1.0;
  for(int i=1;i<=k;i++)
    K = K * (double)i;

  return A/K;

}

double Pk(int n,int k,double p) {
  return Comb(n,k)*pow(1.0-p,n-k)*pow(p,k);
}

double Psk(int n, int k, double p) {
  double sum = 0.0;
  for (int i = k; i <= n; i++) {
    sum += Pk(n,k,p);
  }
  return sum;
}

int _ConvertSMVer2Cores(int major, int minor) {

  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x20, 32}, // Fermi Generation (SM 2.0) GF100 class
      {0x21, 48}, // Fermi Generation (SM 2.1) GF10x class
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {-1, -1} };

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  return 0;

}

GPUEngine::GPUEngine(int nbThreadGroup, int gpuId) {

  // Initialise CUDA
  initialised = false;
  cudaError_t err;

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("GPUEngine: CudaGetDeviceCount %s\n", cudaGetErrorString(error_id));
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  err = cudaSetDevice(gpuId);
  if (err != cudaSuccess) {
    printf("GPUEngine: %s\n", cudaGetErrorString(err));
    return;
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, gpuId);

  if (nbThreadGroup == -1)
    nbThreadGroup = deviceProp.multiProcessorCount * 8;

  nbThread = nbThreadGroup * NB_TRHEAD_PER_GROUP;
  
  char tmp[256];
  sprintf(tmp,"GPU #%d %s (%dx%d cores) Grid(%dx%d)",
  gpuId,deviceProp.name,deviceProp.multiProcessorCount,
  _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
  nbThread / NB_TRHEAD_PER_GROUP,
  NB_TRHEAD_PER_GROUP);
  deviceName = std::string(tmp);
  
  size_t stackSize = 32768;
  err = cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
  if (err != cudaSuccess) {
    printf("GPUEngine: %s\n", cudaGetErrorString(err));
    return;
  }


  /*
  size_t heapSize = ;
  err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(0);
  }

  size_t size;
  cudaDeviceGetLimit(&size, cudaLimitStackSize);
  printf("Stack Size %lld\n", size);
  cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
  printf("Heap Size %lld\n", size);
  */

  // Allocate memory
  err = cudaMalloc((void **)&inputKey, nbThread * 32 * 2);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate input memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&inputKeyPinned, nbThread * 32 * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate input pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaMalloc((void **)&outputPrefix, nbThread * MEMOUT_PER_THREAD);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate output memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&outputPrefixPinned, nbThread * MEMOUT_PER_THREAD, cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate output pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }

  //double P = 1/65536.0;
  //double Plost = Psk(STEP_SIZE,MAX_FOUND,P);
  //printf("Plost=%g\n",Plost);

  searchComp = true;
  initialised = true;

}

int GPUEngine::GetGroupSize() {
  return GRP_SIZE;
}

void GPUEngine::PrintCudaInfo() {

  cudaError_t err;

  const char *sComputeMode[] =
  {
    "Multiple host threads",
    "Only one host thread",
    "No host thread",
    "Multiple process threads",
    "Unknown",
     NULL
  };

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("GPUEngine: CudaGetDeviceCount %s\n", cudaGetErrorString(error_id));
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  for(int i=0;i<deviceCount;i++) {

    err = cudaSetDevice(i);
    if (err != cudaSuccess) {
      printf("GPUEngine: cudaSetDevice(%d) %s\n", i, cudaGetErrorString(err));
      return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB) (%s)\n",
      i,deviceProp.name,deviceProp.multiProcessorCount,
      _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
      deviceProp.major, deviceProp.minor,(double)deviceProp.totalGlobalMem/1048576.0,
      sComputeMode[deviceProp.computeMode]);

  }

}

GPUEngine::~GPUEngine() {

  cudaFree(inputKey);
  cudaFreeHost(outputPrefixPinned);
  cudaFree(outputPrefix);

}

int GPUEngine::GetNbThread() {
  return nbThread;
}

void GPUEngine::SetSearchMode(bool compressed) {
  searchComp = compressed;
}

void GPUEngine::SetPrefix(prefix_t prefix) {
  this->prefix = prefix;
}

bool GPUEngine::callKernel() {

  // Call the kernel (Perform STEP_SIZE keys per thread)
  if(searchComp)
    comp_keys_comp << < nbThread / NB_TRHEAD_PER_GROUP, NB_TRHEAD_PER_GROUP >> > (prefix, inputKey, outputPrefix);
  else
    comp_keys_uncomp << < nbThread / NB_TRHEAD_PER_GROUP, NB_TRHEAD_PER_GROUP >> > (prefix, inputKey, outputPrefix);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: Kernel: %s\n", cudaGetErrorString(err));
    return false;
  }
  return true;

}

bool GPUEngine::SetKeys(Point *p) {

  // Sets the starting keys for each thread
  // p must contains nbThread public keys
  for (int i = 0; i < nbThread; i+= NB_TRHEAD_PER_GROUP) {
    for (int j = 0; j < NB_TRHEAD_PER_GROUP; j++) {

      inputKeyPinned[8*i + j + 0*NB_TRHEAD_PER_GROUP] = p[i + j].x.bits64[0];
      inputKeyPinned[8*i + j + 1*NB_TRHEAD_PER_GROUP] = p[i + j].x.bits64[1];
      inputKeyPinned[8*i + j + 2*NB_TRHEAD_PER_GROUP] = p[i + j].x.bits64[2];
      inputKeyPinned[8*i + j + 3*NB_TRHEAD_PER_GROUP] = p[i + j].x.bits64[3];

      inputKeyPinned[8*i + j + 4*NB_TRHEAD_PER_GROUP] = p[i + j].y.bits64[0];
      inputKeyPinned[8*i + j + 5*NB_TRHEAD_PER_GROUP] = p[i + j].y.bits64[1];
      inputKeyPinned[8*i + j + 6*NB_TRHEAD_PER_GROUP] = p[i + j].y.bits64[2];
      inputKeyPinned[8*i + j + 7*NB_TRHEAD_PER_GROUP] = p[i + j].y.bits64[3];

    }
  }
  // Fill device memory
  cudaMemcpy(inputKey, inputKeyPinned, nbThread*32*2, cudaMemcpyHostToDevice);
  // We do not need the input pinned memory anymore
  cudaFreeHost(inputKeyPinned);
  inputKeyPinned = NULL;

  return callKernel();

}

bool GPUEngine::Launch(std::vector<ITEM> &prefixFound,bool spinWait) {


  prefixFound.clear();

  // Get the result

  if(spinWait) {

    cudaMemcpy(outputPrefixPinned, outputPrefix, MEMOUT_PER_THREAD*nbThread,
      cudaMemcpyDeviceToHost);

  } else {

    // Use cudaMemcpyAsync to avoid default spin wait of cudaMemcpy wich takes 100% CPU
    cudaEvent_t evt;
    cudaEventCreate(&evt);
    cudaMemcpyAsync(outputPrefixPinned, outputPrefix, MEMOUT_PER_THREAD*nbThread,
      cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(evt, 0);
    while (cudaEventQuery(evt) == cudaErrorNotReady) {
      // Sleep 1 ms to free the CPU
#ifdef WIN64
      Sleep(1);
#else
      usleep(1000);
#endif
    }
    cudaEventDestroy(evt);

  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: Launch: %s\n", cudaGetErrorString(err));
    return false;
  }

  // Look for prefix found
  for (int i = 0; i < nbThread; i++) {
    uint32_t thOffset = i * MEMOUT_PER_THREAD;
    uint8_t nbFound = outputPrefixPinned[thOffset];
    for (int j = 0; j < nbFound; j++) {
      uint8_t *itemPtr = outputPrefixPinned + thOffset + 1 + ITEM_SIZE * j;
      ITEM it;
      it.thId = i;
      it.incr = *(uint16_t *)itemPtr;
      it.hash = itemPtr + 2;
      prefixFound.push_back(it);
    }
  }

  return callKernel();

}

std::string toHex(unsigned char *data,int length) {

  string ret;
  char tmp[3];
  for (int i = 0; i < length; i++) {
    if(i && i%4==0) ret.append(" ");
    sprintf(tmp, "%02x", (int)data[i]);
    ret.append(tmp);
  }
  return ret;

}

bool GPUEngine::Check(Secp256K1 &secp) {

  int i = 0;
  int j = 0;
  bool ok = true;

  if(!initialised)
    return false;

  printf("GPU: %s\n",deviceName.c_str());

  // Get endianess
  get_endianness<<<1,1>>>(outputPrefix);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: get_endianness: %s\n", cudaGetErrorString(err));
    return false;
  }
  cudaMemcpy(outputPrefixPinned, outputPrefix,1,cudaMemcpyDeviceToHost);
  littleEndian = *outputPrefixPinned != 0;
  printf("Endianness: %s\n",(littleEndian?"Little":"Big"));

  // Check modular mult
  Int a;
  Int b;
  Int r;
  Int c;
  a.Rand(256);
  b.Rand(256);
  c.ModMulK1(&a,&b);
  memcpy(inputKeyPinned,a.bits64,BIFULLSIZE);
  memcpy(inputKeyPinned+5,b.bits64,BIFULLSIZE);
  cudaMemcpy(inputKey, inputKeyPinned, BIFULLSIZE*2, cudaMemcpyHostToDevice);
  chekc_mult<<<1,1>>>(inputKey,inputKey+5,(uint64_t *)outputPrefix);
  cudaMemcpy(outputPrefixPinned, outputPrefix, BIFULLSIZE, cudaMemcpyDeviceToHost);
  memcpy(r.bits64,outputPrefixPinned,BIFULLSIZE);

  if(!c.IsEqual(&r)) {
    printf("\nModular Mult wrong:\nR=%s\nC=%s\n",
    toHex((uint8_t *)r.bits64,BIFULLSIZE).c_str(),
    toHex((uint8_t *)c.bits64,BIFULLSIZE).c_str());
    return false;
  }

  // Check hash 160
  uint8_t h[20];
  Point pi;
  pi.x.Rand(256);
  pi.y.Rand(256);
  secp.GetHash160(pi, true, h);
  memcpy(inputKeyPinned,pi.x.bits64,BIFULLSIZE);
  memcpy(inputKeyPinned+5,pi.y.bits64,BIFULLSIZE);
  cudaMemcpy(inputKey, inputKeyPinned, BIFULLSIZE*2, cudaMemcpyHostToDevice);
  chekc_hash160<<<1,1>>>(inputKey,inputKey+5,outputPrefix);
  cudaMemcpy(outputPrefixPinned, outputPrefix, 64, cudaMemcpyDeviceToHost);

  if(!ripemd160_comp_hash(outputPrefixPinned,h)) {
    printf("\nGetHask160 wrong:\n%s\n%s\n",
    toHex(outputPrefixPinned,20).c_str(),
    toHex(h,20).c_str());
    return false;
  }

  // Check kernel
  int nbFoundCPU = 0;
  Int k;
  Point *p = new Point[nbThread];
  Point *p2 = new Point[nbThread];
  vector<ITEM> found;

  uint32_t seed = (uint32_t)(Timer::getSeedFromTimer());
  printf("Seed: %u\n",seed);
  rseed(seed);
  for (int i = 0; i < nbThread; i++) {
    k.Rand(256);
    p[i] = secp.ComputePublicKey(&k);
    // Group starts at the middle
    k.Add((uint64_t)GRP_SIZE/2);
    p2[i] = secp.ComputePublicKey(&k);
  }

  SetPrefix(0xFEFE);
  SetKeys(p2);
  double t0 = Timer::get_tick();
  Launch(found,true);
  double t1 = Timer::get_tick();
  Timer::printResult((char *)"Key", STEP_SIZE*nbThread, t0, t1);

  printf("ComputeKeys() found %d items , CPU check:",(int)found.size());

  // Check with CPU
  for (j = 0; (j<nbThread); j++) {
    for (i = 0; i < STEP_SIZE; i++) {
      secp.GetHash160(p[j], searchComp, h);
      prefix_t pr = *(prefix_t *)h;
      if (pr == 0xFEFE) {

	      nbFoundCPU++;

        // Search in found by GPU
        bool f = false;
        int l = 0;
        //printf("Search: %s\n", toHex(h,20).c_str());
        while (l < found.size() && !f) {
          f = ripemd160_comp_hash(found[l].hash,h);
          //printf("[%d]: %s\n", l,toHex(found[l].hash,20).c_str());
          if(!f) l++;
        }
        if (f) {
          found.erase(found.begin() + l);
        } else {
          ok = false;
          printf("\nExpected item not found %s\n", 
            toHex(h,20).c_str());
        }
      }
      p[j] = secp.NextKey(p[j]);
    }
  }

  if (ok && found.size()!=0) {
    ok = false;
    printf("\nUnexpected item found !\n");
  }

  if( !ok ) {
    printf("CPU found %d items\n",nbFoundCPU); 
  }

  if(ok) printf("OK\n");

  delete[] p;
  return ok;

}


