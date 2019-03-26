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
#define UADDO(c, a, b) asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADDC(c, a, b) asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADD(c, a, b) asm volatile ("addc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define UADDO1(c, a) asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADDC1(c, a) asm volatile ("addc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADD1(c, a) asm volatile ("addc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));

#define USUBO(c, a, b) asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUBC(c, a, b) asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUB(c, a, b) asm volatile ("subc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define USUBO1(c, a) asm volatile ("sub.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUBC1(c, a) asm volatile ("subc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUB1(c, a) asm volatile ("subc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) );

#define UMULLO(lo,a, b) asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
#define UMULHI(hi,a, b) asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
#define MADDO(r,a,b,c) asm volatile ("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory" );
#define MADDC(r,a,b,c) asm volatile ("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory" );
#define MADD(r,a,b,c) asm volatile ("madc.hi.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));

__device__ __constant__ uint64_t _0[] = { 0ULL,0ULL,0ULL,0ULL,0ULL };
__device__ __constant__ uint64_t _1[] = { 1ULL,0ULL,0ULL,0ULL,0ULL };

// Field constant (SECPK1)
__device__ __constant__ uint64_t _P[] = { 0xFFFFFFFEFFFFFC2F,0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF,0xFFFFFFFFFFFFFFFF,0ULL };
__device__ __constant__ uint64_t MM64 = 0xD838091DD2253531; // 64bits lsb negative inverse of P (mod 2^64)

__device__ __constant__ uint64_t _beta[] = { 0xC1396C28719501EEULL,0x9CF0497512F58995ULL,0x6E64479EAC3434E9ULL,0x7AE96A2B657C0710ULL };
__device__ __constant__ uint64_t _beta2[] = { 0x3EC693D68E6AFA40ULL,0x630FB68AED0A766AULL,0x919BB86153CBCB16ULL,0x851695D49A83F8EFULL };

#include "GPUGroup.h"
#include "GPUHash.h"

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

  uint64_t t[4];
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

__device__ void ModNeg256(uint64_t *r) {

  uint64_t t[4];
  USUBO(t[0], 0ULL, r[0]);
  USUBC(t[1], 0ULL, r[1]);
  USUBC(t[2], 0ULL, r[2]);
  USUBC(t[3], 0ULL, r[3]);
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

__device__ __noinline__ void _ModInv(uint64_t *R) {

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
  UADD(r[3],r512[3], 0ULL);

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
  UADD(r[3],r512[3], 0ULL);

}

__device__ void _ModSqr(uint64_t *rp, const uint64_t *up) {

  uint64_t r512[8];

  uint64_t u10, u11;

  uint64_t r0;
  uint64_t r1;
  uint64_t r3;
  uint64_t r4;

  uint64_t t1;
  uint64_t t2;


  //k=0
  UMULLO(r512[0], up[0], up[0]);
  UMULHI(r1, up[0], up[0]);

  //k=1
  UMULLO(r3, up[0], up[1]);
  UMULHI(r4, up[0], up[1]);
  UADDO1(r3, r3);
  UADDC1(r4, r4);
  UADD(t1, 0x0ULL, 0x0ULL);
  UADDO1(r3, r1);
  UADDC1(r4, 0x0ULL);
  UADD1(t1, 0x0ULL);
  r512[1] = r3;

  //k=2
  UMULLO(r0, up[0], up[2]);
  UMULHI(r1, up[0], up[2]);
  UADDO1(r0, r0);
  UADDC1(r1, r1);
  UADD(t2, 0x0ULL, 0x0ULL);
  UMULLO(u10, up[1], up[1]);
  UMULHI(u11, up[1], up[1]);
  UADDO1(r0, u10);
  UADDC1(r1, u11);
  UADD1(t2, 0x0ULL);
  UADDO1(r0, r4);
  UADDC1(r1, t1);
  UADD1(t2, 0x0ULL);

  r512[2] = r0;

  //k=3
  UMULLO(r3, up[0], up[3]);
  UMULHI(r4, up[0], up[3]);
  UMULLO(u10, up[1], up[2]);
  UMULHI(u11, up[1], up[2]);
  UADDO1(r3, u10);
  UADDC1(r4, u11);
  UADD(t1, 0x0ULL, 0x0ULL);
  t1 += t1;
  UADDO1(r3, r3);
  UADDC1(r4, r4);
  UADD1(t1, 0x0ULL);
  UADDO1(r3, r1);
  UADDC1(r4, t2);
  UADD1(t1, 0x0ULL);

  r512[3] = r3;

  //k=4
  UMULLO(r0, up[1], up[3]);
  UMULHI(r1, up[1], up[3]);
  UADDO1(r0, r0);
  UADDC1(r1, r1);
  UADD(t2, 0x0ULL, 0x0ULL);
  UMULLO(u10, up[2], up[2]);
  UMULHI(u11, up[2], up[2]);
  UADDO1(r0, u10);
  UADDC1(r1, u11);
  UADD1(t2, 0x0ULL);
  UADDO1(r0, r4);
  UADDC1(r1, t1);
  UADD1(t2, 0x0ULL);

  r512[4] = r0;

  //k=5
  UMULLO(r3, up[2], up[3]);
  UMULHI(r4, up[2], up[3]);
  UADDO1(r3, r3);
  UADDC1(r4, r4);
  UADD(t1, 0x0ULL, 0x0ULL);
  UADDO1(r3, r1);
  UADDC1(r4, t2);
  UADD1(t1, 0x0ULL);

  r512[5] = r3;

  //k=6
  UMULLO(r0, up[3], up[3]);
  UMULHI(r1, up[3], up[3]);
  UADDO1(r0, r4);
  UADD1(r1, t1);
  r512[6] = r0;

  //k=7
  r512[7] = r1;

#if 1

  // Reduce from 512 to 320 
  UMULLO(r0, r512[4], 0x1000003D1ULL);
  UMULLO(r1, r512[5], 0x1000003D1ULL);
  MADDO(r1, r512[4], 0x1000003D1ULL, r1);
  UMULLO(t2, r512[6], 0x1000003D1ULL);
  MADDC(t2, r512[5], 0x1000003D1ULL, t2);
  UMULLO(r3, r512[7], 0x1000003D1ULL);
  MADDC(r3, r512[6], 0x1000003D1ULL, r3);
  MADD(r4, r512[7], 0x1000003D1ULL, 0ULL);

  UADDO1(r512[0], r0);
  UADDC1(r512[1], r1);
  UADDC1(r512[2], t2);
  UADDC1(r512[3], r3);

  // Reduce from 320 to 256
  UADD1(r4, 0ULL);
  UMULLO(u10, r4, 0x1000003D1ULL);
  UMULHI(u11, r4, 0x1000003D1ULL);
  UADDO(rp[0], r512[0], u10);
  UADDC(rp[1], r512[1], u11);
  UADDC(rp[2], r512[2], 0ULL);
  UADD(rp[3], r512[3], 0ULL);

#else
  
  uint64_t z1, z2, z3, z4, z5, z6, z7, z8;

  UMULLO(z3, r512[5], 0x1000003d1ULL);
  UMULHI(z4, r512[5], 0x1000003d1ULL);
  UMULLO(z5, r512[6], 0x1000003d1ULL);
  UMULHI(z6, r512[6], 0x1000003d1ULL);
  UMULLO(z7, r512[7], 0x1000003d1ULL);
  UMULHI(z8, r512[7], 0x1000003d1ULL);
  UMULLO(z1, r512[4], 0x1000003d1ULL);
  UMULHI(z2, r512[4], 0x1000003d1ULL);
  UADDO1(z1, r512[0]);
  UADD1(z2, 0x0ULL);


  UADDO1(z2, r512[1]);
  UADDC1(z4, r512[2]);
  UADDC1(z6, r512[3]);
  UADD1(z8, 0x0ULL);

  UADDO1(z3, z2);
  UADDC1(z5, z4);
  UADDC1(z7, z6);
  UADD1(z8, 0x0ULL);

  UMULLO(u10, z8, 0x1000003d1ULL);
  UMULHI(u11, z8, 0x1000003d1ULL);
  UADDO1(z1, u10);
  UADDC1(z3, u11);
  UADDC1(z5, 0x0ULL);
  UADD1(z7, 0x0ULL);

  rp[0] = z1;
  rp[1] = z3;
  rp[2] = z5;
  rp[3] = z7;

#endif

}

// ---------------------------------------------------------------------------------------
// Compute all ModInv of the group
// ---------------------------------------------------------------------------------------

__device__ __noinline__ void _ModInvGrouped(uint64_t r[GRP_SIZE / 2 + 1][4]) {

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
// EC
// ---------------------------------------------------------------------------------

__device__ __noinline__ void _GetHash160Comp(uint64_t *x, uint8_t isOdd, uint8_t *hash) {

  uint32_t *x32 = (uint32_t *)(x);
  uint32_t publicKeyBytes[16];
  uint32_t s[16];

  // Compressed public key
  publicKeyBytes[0] = __byte_perm(x32[7], 0x2 + isOdd, 0x4321 );
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

__device__ __noinline__ void _GetHash160(uint64_t *x, uint64_t *y, uint8_t *hash) {

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

#include "GPUCompute.h"

// ---------------------------------------------------------------------------------------

__global__ void comp_keys(uint32_t mode,prefix_t *prefix, uint32_t *lookup32, uint64_t *keys, uint32_t maxFound, uint32_t *found) {

  int xPtr = (blockIdx.x*blockDim.x) * 8;
  int yPtr = xPtr + 4 * NB_TRHEAD_PER_GROUP;
  ComputeKeys(mode, keys + xPtr, keys + yPtr, prefix, lookup32, maxFound, found);

}

__global__ void comp_keys_comp(prefix_t *prefix, uint32_t *lookup32, uint64_t *keys, uint32_t maxFound, uint32_t *found) {

  int xPtr = (blockIdx.x*blockDim.x) * 8;
  int yPtr = xPtr + 4 * NB_TRHEAD_PER_GROUP;
  ComputeKeysComp(keys + xPtr, keys + yPtr, prefix, lookup32, maxFound, found);

}

//#define FULLCHECK
#ifdef FULLCHECK

// ---------------------------------------------------------------------------------------

__global__ void chekc_mult(uint64_t *a, uint64_t *b, uint64_t *r) {

  _ModMult(r, a, b);
  r[4]=0;

}

// ---------------------------------------------------------------------------------------

__global__ void chekc_hash160(uint64_t *x, uint64_t *y, uint32_t *h) {

  _GetHash160(x, y, (uint8_t *)h);
  _GetHash160Comp(x, y, (uint8_t *)(h+5));

}

// ---------------------------------------------------------------------------------------

__global__ void get_endianness(uint32_t *endian) {

  uint32_t a = 0x01020304;
  uint8_t fb = *(uint8_t *)(&a);
  *endian = (fb==0x04);

}

#endif //FULLCHECK

// ---------------------------------------------------------------------------------------

using namespace std;

std::string toHex(unsigned char *data, int length) {

  string ret;
  char tmp[3];
  for (int i = 0; i < length; i++) {
    if (i && i % 4 == 0) ret.append(" ");
    sprintf(tmp, "%02x", (int)data[i]);
    ret.append(tmp);
  }
  return ret;

}

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

GPUEngine::GPUEngine(int nbThreadGroup, int gpuId, uint32_t maxFound) {

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

  this->nbThread = nbThreadGroup * NB_TRHEAD_PER_GROUP;
  this->maxFound = maxFound;
  this->outputSize = (maxFound*ITEM_SIZE + 4);

  char tmp[512];
  sprintf(tmp,"GPU #%d %s (%dx%d cores) Grid(%dx%d)",
  gpuId,deviceProp.name,deviceProp.multiProcessorCount,
  _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
  nbThread / NB_TRHEAD_PER_GROUP,
  NB_TRHEAD_PER_GROUP);
  deviceName = std::string(tmp);

  // Prefer L1 (We do not use __shared__ at all)
  err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  if (err != cudaSuccess) {
    printf("GPUEngine: %s\n", cudaGetErrorString(err));
    return;
  }

  size_t stackSize = 49152;
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
  err = cudaMalloc((void **)&inputPrefix, _64K * 2);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate prefix memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&inputPrefixPinned, _64K * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate prefix pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }
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
  err = cudaMalloc((void **)&outputPrefix, outputSize);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate output memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&outputPrefixPinned, outputSize, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate output pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }

  searchMode = SEARCH_COMPRESSED;
  initialised = true;
  inputPrefixLookUp = NULL;

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
  cudaFree(inputPrefix);
  if(inputPrefixLookUp) cudaFree(inputPrefixLookUp);
  cudaFreeHost(outputPrefixPinned);
  cudaFree(outputPrefix);

}

int GPUEngine::GetNbThread() {
  return nbThread;
}

void GPUEngine::SetSearchMode(int searchMode) {
  this->searchMode = searchMode;
}

void GPUEngine::SetPrefix(std::vector<prefix_t> prefixes) {

  memset(inputPrefixPinned, 0, _64K * 2);
  for(int i=0;i<(int)prefixes.size();i++)
    inputPrefixPinned[prefixes[i]]=1;

  // Fill device memory
  cudaMemcpy(inputPrefix, inputPrefixPinned, _64K * 2, cudaMemcpyHostToDevice);

  // We do not need the input pinned memory anymore
  cudaFreeHost(inputPrefixPinned);
  inputPrefixPinned = NULL;
  lostWarning = false;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: SetPrefix: %s\n", cudaGetErrorString(err));
  }

}

void GPUEngine::SetPrefix(std::vector<LPREFIX> prefixes, uint32_t totalPrefix) {

  // Allocate memory for the second level of lookup tables
  cudaError_t err = cudaMalloc((void **)&inputPrefixLookUp, (_64K+totalPrefix) * 4);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate prefix lookup memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&inputPrefixLookUpPinned, (_64K+totalPrefix) * 4, cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate prefix lookup pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }

  uint32_t offset = _64K;
  memset(inputPrefixPinned, 0, _64K * 2);
  memset(inputPrefixLookUpPinned, 0, _64K * 4);
  for (int i = 0; i < (int)prefixes.size(); i++) {
    int nbLPrefix = (int)prefixes[i].lPrefixes.size();
    inputPrefixPinned[prefixes[i].sPrefix] = (uint16_t)nbLPrefix;
    inputPrefixLookUpPinned[prefixes[i].sPrefix] = offset;
    for (int j = 0; j < nbLPrefix; j++) {
      inputPrefixLookUpPinned[offset++]=prefixes[i].lPrefixes[j];
    }
  }

  if (offset != (_64K+totalPrefix)) {
    printf("GPUEngine: Wrong totalPrefix %d!=%d!\n",offset- _64K, totalPrefix);
    return;
  }

  // Fill device memory
  cudaMemcpy(inputPrefix, inputPrefixPinned, _64K * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(inputPrefixLookUp, inputPrefixLookUpPinned, (_64K+totalPrefix) * 4, cudaMemcpyHostToDevice);

  // We do not need the input pinned memory anymore
  cudaFreeHost(inputPrefixPinned);
  inputPrefixPinned = NULL;
  cudaFreeHost(inputPrefixLookUpPinned);
  inputPrefixLookUpPinned = NULL;
  lostWarning = false;

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: SetPrefix (large): %s\n", cudaGetErrorString(err));
  }

}
bool GPUEngine::callKernel() {

  // Reset nbFound
  cudaMemset(outputPrefix,0,4);

  // Call the kernel (Perform STEP_SIZE keys per thread)
  if( searchMode==SEARCH_COMPRESSED ) {
    comp_keys_comp<<< nbThread / NB_TRHEAD_PER_GROUP, NB_TRHEAD_PER_GROUP >>>
        (inputPrefix, inputPrefixLookUp, inputKey, maxFound, outputPrefix);
  } else {
    comp_keys<<< nbThread / NB_TRHEAD_PER_GROUP, NB_TRHEAD_PER_GROUP >>>
        (searchMode, inputPrefix, inputPrefixLookUp, inputKey, maxFound, outputPrefix);
  }

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

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: SetKeys: %s\n", cudaGetErrorString(err));
  }

  return callKernel();

}

bool GPUEngine::Launch(std::vector<ITEM> &prefixFound,bool spinWait) {


  prefixFound.clear();

  // Get the result

  if(spinWait) {

    cudaMemcpy(outputPrefixPinned, outputPrefix, outputSize, cudaMemcpyDeviceToHost);

  } else {

    // Use cudaMemcpyAsync to avoid default spin wait of cudaMemcpy wich takes 100% CPU
    cudaEvent_t evt;
    cudaEventCreate(&evt);
    cudaMemcpyAsync(outputPrefixPinned, outputPrefix, 4, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(evt, 0);
    while (cudaEventQuery(evt) == cudaErrorNotReady) {
      // Sleep 1 ms to free the CPU
      Timer::SleepMillis(1);
    }
    cudaEventDestroy(evt);

  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: Launch: %s\n", cudaGetErrorString(err));
    return false;
  }

  // Look for prefix found
  uint32_t nbFound = outputPrefixPinned[0];
  if (nbFound > maxFound) {
    // prefix has been lost
    if (!lostWarning) {
      printf("\nWarning, %d items lost\nHint: Search with less prefixes, less threads (-g) or increase maxFound (-m)\n", (nbFound - maxFound));
      lostWarning = true;
    }
    nbFound = maxFound;
  }
  
  // When can perform a standard copy, the kernel is eneded
  cudaMemcpy( outputPrefixPinned , outputPrefix , nbFound*ITEM_SIZE + 4 , cudaMemcpyDeviceToHost);
  
  for (uint32_t i = 0; i < nbFound; i++) {
    uint32_t *itemPtr = outputPrefixPinned + (i*ITEM_SIZE32 + 1);
    ITEM it;
    it.thId = itemPtr[0];
    int16_t *ptr = (int16_t *)&(itemPtr[1]);
    it.endo = ptr[0] & 0x7FFF;
    it.mode = (ptr[0]&0x8000)!=0;
    it.incr = ptr[1];
    it.hash = (uint8_t *)(itemPtr + 2);
    prefixFound.push_back(it);
  }

  return callKernel();

}

bool GPUEngine::CheckHash(uint8_t *h, vector<ITEM>& found,int tid,int incr,int endo, int *nbOK) {

  bool ok = true;

  // Search in found by GPU
  bool f = false;
  int l = 0;
  //printf("Search: %s\n", toHex(h,20).c_str());
  while (l < found.size() && !f) {
    f = ripemd160_comp_hash(found[l].hash, h);
    if (!f) l++;
  }
  if (f) {
    found.erase(found.begin() + l);
    *nbOK = *nbOK+1;
  } else {
    ok = false;
    printf("Expected item not found %s (thread=%d, incr=%d, endo=%d)\n",
      toHex(h, 20).c_str(),tid,incr,endo);
  }

  return ok;

}

bool GPUEngine::Check(Secp256K1 &secp) {

  uint8_t h[20];
  int i = 0;
  int j = 0;
  bool ok = true;

  if(!initialised)
    return false;

  printf("GPU: %s\n",deviceName.c_str());

#ifdef FULLCHECK

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

  // Check hash 160C
  uint8_t hc[20];
  Point pi;
  pi.x.Rand(256);
  pi.y.Rand(256);
  secp.GetHash160(pi, false, h);
  secp.GetHash160(pi, true, hc);
  memcpy(inputKeyPinned,pi.x.bits64,BIFULLSIZE);
  memcpy(inputKeyPinned+5,pi.y.bits64,BIFULLSIZE);
  cudaMemcpy(inputKey, inputKeyPinned, BIFULLSIZE*2, cudaMemcpyHostToDevice);
  chekc_hash160<<<1,1>>>(inputKey,inputKey+5,outputPrefix);
  cudaMemcpy(outputPrefixPinned, outputPrefix, 64, cudaMemcpyDeviceToHost);

  if(!ripemd160_comp_hash((uint8_t *)outputPrefixPinned,h)) {
    printf("\nGetHask160 wrong:\n%s\n%s\n",
    toHex((uint8_t *)outputPrefixPinned,20).c_str(),
    toHex(h,20).c_str());
    return false;
  }
  if (!ripemd160_comp_hash((uint8_t *)(outputPrefixPinned+5), hc)) {
    printf("\nGetHask160Comp wrong:\n%s\n%s\n",
      toHex((uint8_t *)(outputPrefixPinned + 5), 20).c_str(),
      toHex(h, 20).c_str());
    return false;
  }

#endif //FULLCHECK


  // Check kernel
  int nbFoundCPU[6];
  int nbOK[6];
  Int k;
  Point *p = new Point[nbThread];
  Point *p2 = new Point[nbThread];
  vector<ITEM> found;
  bool searchComp;

  if (searchMode == SEARCH_BOTH) {
    printf("Warning, Check function does not support BOTH_MODE, use either compressed or uncompressed");
    return true;
  }

  searchComp = (searchMode == SEARCH_COMPRESSED)?true:false;

  uint32_t seed = (uint32_t)(Timer::getSeedFromTimer());
  printf("Seed: %u\n",seed);
  rseed(seed);
  memset(nbOK,0,sizeof(nbOK));
  memset(nbFoundCPU, 0, sizeof(nbFoundCPU));
  for (int i = 0; i < nbThread; i++) {
    k.Rand(256);
    p[i] = secp.ComputePublicKey(&k);
    // Group starts at the middle
    k.Add((uint64_t)GRP_SIZE/2);
    p2[i] = secp.ComputePublicKey(&k);
  }

  std::vector<prefix_t> prefs;
  prefs.push_back(0xFEFE);
  prefs.push_back(0x1234);
  SetPrefix(prefs);
  SetKeys(p2);
  double t0 = Timer::get_tick();
  Launch(found,true);
  double t1 = Timer::get_tick();
  Timer::printResult((char *)"Key", 6*STEP_SIZE*nbThread, t0, t1);
   
  //for (int i = 0; i < found.size(); i++) {
  //  printf("[%d]: thId=%d incr=%d\n", i, found[i].thId,found[i].incr);
  //  printf("[%d]: %s\n", i,toHex(found[i].hash,20).c_str());
  //}
  
  printf("ComputeKeys() found %d items , CPU check...\n",(int)found.size());

  Int beta,beta2;
  beta.SetBase16((char *)"7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee");
  beta2.SetBase16((char *)"851695d49a83f8ef919bb86153cbcb16630fb68aed0a766a3ec693d68e6afa40");

  // Check with CPU
  for (j = 0; (j<nbThread); j++) {
    for (i = 0; i < STEP_SIZE; i++) {
      
      Point pt,p1,p2;
      pt = p[j];
      p1 = p[j];
      p2 = p[j];
      p1.x.ModMulK1(&beta);
      p2.x.ModMulK1(&beta2);
      p[j] = secp.NextKey(p[j]);

      // Point and endo
      secp.GetHash160(pt, searchComp, h);
      prefix_t pr = *(prefix_t *)h;
      if (pr == 0xFEFE || pr == 0x1234) {
	      nbFoundCPU[0]++;
        ok &= CheckHash(h,found, j, i, 0, nbOK + 0);
      }
      secp.GetHash160(p1, searchComp, h);
      pr = *(prefix_t *)h;
      if (pr == 0xFEFE || pr == 0x1234) {
        nbFoundCPU[1]++;
        ok &= CheckHash(h, found, j, i, 1, nbOK + 1);
      }
      secp.GetHash160(p2, searchComp, h);
      pr = *(prefix_t *)h;
      if (pr == 0xFEFE || pr == 0x1234) {
        nbFoundCPU[2]++;
        ok &= CheckHash(h, found, j, i, 2, nbOK + 2);
      }

      // Symetrics
      pt.y.ModNeg();
      p1.y.ModNeg();
      p2.y.ModNeg();

      secp.GetHash160(pt, searchComp, h);
      pr = *(prefix_t *)h;
      if (pr == 0xFEFE || pr == 0x1234) {
        nbFoundCPU[3]++;
        ok &= CheckHash(h, found, j, -i, 0, nbOK + 3);
      }
      secp.GetHash160(p1, searchComp, h);
      pr = *(prefix_t *)h;
      if (pr == 0xFEFE || pr == 0x1234) {
        nbFoundCPU[4]++;
        ok &= CheckHash(h, found, j, -i, 1, nbOK + 4);
      }
      secp.GetHash160(p2, searchComp, h);
      pr = *(prefix_t *)h;
      if (pr == 0xFEFE || pr == 0x1234) {
        nbFoundCPU[5]++;
        ok &= CheckHash(h, found, j, -i, 2, nbOK + 5);
      }

    }
  }

  if (ok && found.size()!=0) {
    ok = false;
    printf("Unexpected item found !\n");
  }

  if( !ok ) {

    int nbF = nbFoundCPU[0] + nbFoundCPU[1] + nbFoundCPU[2] +
              nbFoundCPU[3] + nbFoundCPU[4] + nbFoundCPU[5];
    printf("CPU found %d items\n",nbF); 

    printf("GPU: point   correct [%d/%d]\n", nbOK[0] , nbFoundCPU[0]);
    printf("GPU: endo #1 correct [%d/%d]\n", nbOK[1] , nbFoundCPU[1]);
    printf("GPU: endo #2 correct [%d/%d]\n", nbOK[2] , nbFoundCPU[2]);

    printf("GPU: sym/point   correct [%d/%d]\n", nbOK[3] , nbFoundCPU[3]);
    printf("GPU: sym/endo #1 correct [%d/%d]\n", nbOK[4] , nbFoundCPU[4]);
    printf("GPU: sym/endo #2 correct [%d/%d]\n", nbOK[5] , nbFoundCPU[5]);

    printf("GPU/CPU check Failed !\n");

  }

  if(ok) printf("GPU/CPU check OK\n");

  delete[] p;
  return ok;

}


