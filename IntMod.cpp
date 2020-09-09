/*
 * This file is part of the BSGS distribution (https://github.com/JeanLucPons/VanitySearch).
 * Copyright (c) 2020 Jean Luc PONS.
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

#include "Int.h"
#include <emmintrin.h>
#include <string.h>

#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

static Int     _P;       // Field characteristic
static Int     _R;       // Montgomery multiplication R
static Int     _R2;      // Montgomery multiplication R2
static Int     _R3;      // Montgomery multiplication R3
static Int     _R4;      // Montgomery multiplication R4
static int32_t  Msize;    // Montgomery mult size
static uint32_t MM32;     // 32bits lsb negative inverse of P
static uint64_t MM64;     // 64bits lsb negative inverse of P
#define MSK62  0x3FFFFFFFFFFFFFFF

extern Int _ONE;

// ------------------------------------------------

void Int::ModAdd(Int *a) {
  Int p;
  Add(a);
  p.Sub(this,&_P);
  if(p.IsPositive())
    Set(&p);
}

// ------------------------------------------------

void Int::ModAdd(Int *a, Int *b) {
  Int p;
  Add(a,b);
  p.Sub(this,&_P);
  if(p.IsPositive())
    Set(&p);
}

// ------------------------------------------------

void Int::ModDouble() {
  Int p;
  Add(this);
  p.Sub(this,&_P);
  if(p.IsPositive())
    Set(&p);
}

// ------------------------------------------------

void Int::ModAdd(uint64_t a) {
  Int p;
  Add(a);
  p.Sub(this,&_P);
  if(p.IsPositive())
    Set(&p);
}

// ------------------------------------------------

void Int::ModSub(Int *a) {
  Sub(a);
  if (IsNegative())
    Add(&_P);
}

// ------------------------------------------------

void Int::ModSub(uint64_t a) {
  Sub(a);
  if (IsNegative())
    Add(&_P);
}

// ------------------------------------------------

void Int::ModSub(Int *a,Int *b) {
  Sub(a,b);
  if (IsNegative())
    Add(&_P);
}

// ------------------------------------------------

void Int::ModNeg() {
  Neg();
  Add(&_P);
}

// ------------------------------------------------

// INV256[x] = x^-1 (mod 256)
int64_t INV256[] = {
    -0LL,-1LL,-0LL,-235LL,-0LL,-141LL,-0LL,-183LL,-0LL,-57LL,-0LL,-227LL,-0LL,-133LL,-0LL,-239LL,
    -0LL,-241LL,-0LL,-91LL,-0LL,-253LL,-0LL,-167LL,-0LL,-41LL,-0LL,-83LL,-0LL,-245LL,-0LL,-223LL,
    -0LL,-225LL,-0LL,-203LL,-0LL,-109LL,-0LL,-151LL,-0LL,-25LL,-0LL,-195LL,-0LL,-101LL,-0LL,-207LL,
    -0LL,-209LL,-0LL,-59LL,-0LL,-221LL,-0LL,-135LL,-0LL,-9LL,-0LL,-51LL,-0LL,-213LL,-0LL,-191LL,
    -0LL,-193LL,-0LL,-171LL,-0LL,-77LL,-0LL,-119LL,-0LL,-249LL,-0LL,-163LL,-0LL,-69LL,-0LL,-175LL,
    -0LL,-177LL,-0LL,-27LL,-0LL,-189LL,-0LL,-103LL,-0LL,-233LL,-0LL,-19LL,-0LL,-181LL,-0LL,-159LL,
    -0LL,-161LL,-0LL,-139LL,-0LL,-45LL,-0LL,-87LL,-0LL,-217LL,-0LL,-131LL,-0LL,-37LL,-0LL,-143LL,
    -0LL,-145LL,-0LL,-251LL,-0LL,-157LL,-0LL,-71LL,-0LL,-201LL,-0LL,-243LL,-0LL,-149LL,-0LL,-127LL,
    -0LL,-129LL,-0LL,-107LL,-0LL,-13LL,-0LL,-55LL,-0LL,-185LL,-0LL,-99LL,-0LL,-5LL,-0LL,-111LL,
    -0LL,-113LL,-0LL,-219LL,-0LL,-125LL,-0LL,-39LL,-0LL,-169LL,-0LL,-211LL,-0LL,-117LL,-0LL,-95LL,
    -0LL,-97LL,-0LL,-75LL,-0LL,-237LL,-0LL,-23LL,-0LL,-153LL,-0LL,-67LL,-0LL,-229LL,-0LL,-79LL,
    -0LL,-81LL,-0LL,-187LL,-0LL,-93LL,-0LL,-7LL,-0LL,-137LL,-0LL,-179LL,-0LL,-85LL,-0LL,-63LL,
    -0LL,-65LL,-0LL,-43LL,-0LL,-205LL,-0LL,-247LL,-0LL,-121LL,-0LL,-35LL,-0LL,-197LL,-0LL,-47LL,
    -0LL,-49LL,-0LL,-155LL,-0LL,-61LL,-0LL,-231LL,-0LL,-105LL,-0LL,-147LL,-0LL,-53LL,-0LL,-31LL,
    -0LL,-33LL,-0LL,-11LL,-0LL,-173LL,-0LL,-215LL,-0LL,-89LL,-0LL,-3LL,-0LL,-165LL,-0LL,-15LL,
    -0LL,-17LL,-0LL,-123LL,-0LL,-29LL,-0LL,-199LL,-0LL,-73LL,-0LL,-115LL,-0LL,-21LL,-0LL,-255LL, };

void Int::DivStep62(Int* u,Int* v,int64_t* eta,int* pos,int64_t* uu,int64_t* uv,int64_t* vu,int64_t* vv) {

  // u' = (uu*u + uv*v) >> bitCount
  // v' = (vu*u + vv*v) >> bitCount
  // Do not maintain a matrix for r and s, the number of 
  // 'added P' can be easily calculated
  // Performance are measured on a I5-8500 for P=2^256 - 0x1000003D1 (VS2019 compilation)

  int bitCount;
  uint64_t u0 = u->bits64[0];
  uint64_t v0 = v->bits64[0];

#if 0

  *uu = 1; *uv = 0;
  *vu = 0; *vv = 1;

  #define SWAP_ADD(x,y) x+=y;y-=x;
  #define SWAP_SUB(x,y) x-=y;y+=x;

  // Former divstep62 (using __builtin_ctzll)
  // Avg: 632 Kinv/s, Avg number of divstep62: 9.83

  bitCount = 62;
  int64_t nb0;
  __m128i _u;
  __m128i _v;
  _u.m128i_u64[0] = 1;
  _u.m128i_u64[1] = 0;
  _v.m128i_u64[0] = 0;
  _v.m128i_u64[1] = 1;

  while(true) {

    int zeros = TZC(v0 | (UINT64_MAX << bitCount));
    v0 >>= zeros;
    _u = _mm_slli_epi64(_u,(int)zeros);
    bitCount -= zeros;

    if(bitCount <= 0)
      break;

    nb0 = (v0 + u0) & 0x3;
    if(nb0 == 0) {
      _v = _mm_add_epi64(_v,_u);
      _u = _mm_sub_epi64(_u,_v);
      SWAP_ADD(v0,u0);
    } else {
      _v = _mm_sub_epi64(_v,_u);
      _u = _mm_add_epi64(_u,_v);
      SWAP_SUB(v0,u0);
    }

  }
  *uu = _u.m128i_u64[0];
  *uv = _u.m128i_u64[1];
  *vu = _v.m128i_u64[0];
  *vv = _v.m128i_u64[1];

#endif

#if 1

  #define SWAP(tmp,x,y) tmp = x; x = y; y = tmp;

  // divstep62 var time implementation (Thomas Pornin's method)
  // (see https://github.com/pornin/bingcd)
  // Avg 780 Kinv/s, Avg number of divstep62: 6.13
  // "Make u,v positive" in the macro loop must be enabled

  uint64_t uh;
  uint64_t vh;
  uint64_t w,x;
  unsigned char c = 0;

  // Extract 64 MSB of u and v
  // u and v must be positive

  while(*pos>=1 && (u->bits64[*pos] | v->bits64[*pos])==0) (*pos)--;
  if(*pos==0) {
    uh = u->bits64[0];
    vh = v->bits64[0];
  } else {
    uint64_t s = LZC(u->bits64[*pos] | v->bits64[*pos]);
    if(s == 0) {
      uh = u->bits64[*pos];
      vh = v->bits64[*pos];
    } else {
      uh = __shiftleft128(u->bits64[*pos-1],u->bits64[*pos],(uint8_t)s);
      vh = __shiftleft128(v->bits64[*pos-1],v->bits64[*pos],(uint8_t)s);
    }
  }

  bitCount = 62;

  __m128i _u;
  __m128i _v;
  __m128i _t;

#ifdef WIN64
  _u.m128i_u64[0] = 1;
  _u.m128i_u64[1] = 0;
  _v.m128i_u64[0] = 0;
  _v.m128i_u64[1] = 1;
#else
  ((int64_t *)&_u)[0] = 1;
  ((int64_t *)&_u)[1] = 0;
  ((int64_t *)&_v)[0] = 0;
  ((int64_t *)&_v)[1] = 1;
#endif

  while(true) {

    // Use a sentinel bit to count zeros only up to bitCount
    uint64_t zeros = TZC(v0 | 1ULL << bitCount);
    vh >>= zeros;
    v0 >>= zeros;
    _u = _mm_slli_epi64(_u,(int)zeros);
    bitCount -= (int)zeros;

    if(bitCount <= 0) {
      break;
    }

    if( vh < uh ) {
      SWAP(w,uh,vh);
      SWAP(x,u0,v0);
      SWAP(_t,_u,_v);
    }

    vh -= uh;
    v0 -= u0;
    _v = _mm_sub_epi64(_v,_u);

  }

#ifdef WIN64
  *uu = _u.m128i_u64[0];
  *uv = _u.m128i_u64[1];
  *vu = _v.m128i_u64[0];
  *vv = _v.m128i_u64[1];
#else
  *uu = ((int64_t *)&_u)[0];
  *uv = ((int64_t *)&_u)[1];
  *vu = ((int64_t *)&_v)[0];
  *vv = ((int64_t *)&_v)[1];
#endif

#endif

#if 0

  #define SWAP_NEG(tmp,x,y) tmp = x; x = y; y = -tmp;

  int64_t m,w,x,y,z;
  bitCount = 62;
  int64_t limit;

  *uu = 1; *uv = 0;
  *vu = 0; *vv = 1;

  // divstep62 var time implementation by Peter Dettman (based on Bernstein/Yang paper)
  // (see https://github.com/bitcoin-core/secp256k1/pull/767)
  // Avg: 700 Kinv/s, Avg number of divstep62: 9.00

  while(true) {

    // Use a sentinel bit to count zeros only up to bitCount
    int zeros = TZC(v0 | (1ULL << bitCount));

    v0 >>= zeros;
    *uu <<= zeros;
    *uv <<= zeros;
    *eta -= zeros;
    bitCount -= zeros;

    if(bitCount <= 0) {
      break;
    }

    if(*eta < 0) {
      *eta = -*eta;
      SWAP_NEG(x,u0,v0);
      SWAP_NEG(y,*uu,*vu);
      SWAP_NEG(z,*uv,*vv);
    }

    // Handle up to 6 divstep at once
    limit = (*eta + 1) > bitCount ? bitCount : (*eta + 1);
    m = (UINT64_MAX >> (64 - limit)) & 63U;
    //w = (u0 * v0 * (u0 * u0 - 2)) & m; // w = v0 * -u0^-1 mod 2^6  (1 Newton step => 6bit)
    w = (v0 * INV256[u0 & 63U]) & m;


    v0 += u0 * w;
    *vu += *uu * w;
    *vv += *uv * w;

  }

#endif

#if 0

  // divstep62 constant time implementation by Peter Dettman (based on Bernstein/Yang paper)
  // (see https://github.com/bitcoin-core/secp256k1/pull/767)
  // Avg: 405 Kinv/s, Avg number of divstep62: 9.00
  uint64_t c1,c2,x,y,z;

  for(bitCount = 0; bitCount < 62; bitCount++) {

    c1 = -(v0 & ((uint64_t)(*eta) >> 63));

    x = (u0 ^ v0) & c1;
    u0 ^= x; v0 ^= x; v0 ^= c1; v0 -= c1;

    y = (*uu ^ *vu) & c1;
    *uu ^= y; *vu ^= y; *vu ^= c1; *vu -= c1;

    z = (*uv ^ *vv) & c1;
    *uv ^= z; *vv ^= z; *vv ^= c1; *vv -= c1;

    *eta = (*eta ^ c1) - c1 - 1;

    c2 = -(v0 & 1);

    v0 += (u0 & c2); v0 >>= 1;
    *vu += (*uu & c2); *uu <<= 1;
    *vv += (*uv & c2); *uv <<= 1;
  }

#endif

}

// ------------------------------------------------

uint64_t totalCount;

void Int::ModInv() {

  // Compute modular inverse of this mop _P
  // 0 <= this < _P  , _P must be odd
  // Return 0 if no inverse

  // 256bit 
  //#define XCD 1               // ~97  kOps/s
  //#define MONTGOMERY 1        // ~360 kOps/s
  #define DRS62 1                // ~780 kOps/s

  Int u(&_P);
  Int v(this);
  Int r((int64_t)0);
  Int s((int64_t)1);

#ifdef XCD

  Int q, t1, t2, w;

  // Classic XCD 

  bool bIterations = true;  // Remember odd/even iterations
  while (!u.IsZero()) {
    // Step X3. Divide and "Subtract"
    q.Set(&v);
    q.Div(&u, &t2);   // q = u / v, t2 = u % v
    w.Mult(&q, &r);   // w = q * r
    t1.Add(&s, &w);   // t1 = s + w

                      // Swap u,v & r,s
    s.Set(&r);
    r.Set(&t1);
    v.Set(&u);
    u.Set(&t2);

    bIterations = !bIterations;
  }

  if (!v.IsOne()) {
    CLEAR();
    return;
  }

  if (!bIterations) {
    Set(&_P);
    Sub(&s);  /* inv = n - u1 */
  } else {
    Set(&s);  /* inv = u1     */
  }

#endif

#ifdef MONTGOMERY

  Int x;
  int k = 0;
  if(v.IsZero()) {
    Set(&v);
    return;
  }

  // Montgomery method
  while (v.IsStrictPositive()) {
    if (u.IsEven()) {
      shiftR(1, u.bits64);
      shiftL(1, s.bits64);
    } else if (v.IsEven()) {
      shiftR(1, v.bits64);
      shiftL(1, r.bits64);
    } else {
      x.Set(&u);
      x.Sub(&v);
      if (x.IsStrictPositive()) {
        shiftR(1, x.bits64);
        u.Set(&x);
        r.Add(&s);
        shiftL(1, s.bits64);
      } else {
        x.Neg();
        shiftR(1, x.bits64);
        v.Set(&x);
        s.Add(&r);
        shiftL(1, r.bits64);
      }
    }
    k++;
  }

  if (r.IsGreater(&_P))
    r.Sub(&_P);
  r.Neg();
  r.Add(&_P);

  // Demontgomerise (divide by 2^k)
  uint64_t ML;
  uint64_t carryR;
  while (k>=64) {
    ML = r.bits64[0] * MM64;
    imm_umul(_P.bits64,ML,s.bits64);
    carryR = r.AddCh(&s,0);
    r.ShiftR64Bit();
    r.bits64[NB64BLOCK-1] = carryR;
    k-=64;
  }
  if(k>0) {
    uint64_t mask = (1ULL << k) - 1;
    ML = (r.bits64[0] * MM64) & mask;
    imm_umul(_P.bits64,ML,s.bits64);
    carryR = r.AddCh(&s,0);
    shiftR(k,r.bits64,carryR);
  }
  if(r.IsGreater(&_P))
    r.Sub(&_P);

  Set(&r);

#endif

#ifdef DRS62

  // Delayed right shift 62bits
  Int r0_P;
  Int s0_P;

  int64_t  eta = -1;
  int64_t uu,uv,vu,vv;
  uint64_t carryS,carryR;
  int pos = NB64BLOCK - 1;
  while(pos >= 1 && (u.bits64[pos] | v.bits64[pos]) == 0) pos--;

  //printf("ModInv(%s)\n",GetBase16().c_str());

  while (!v.IsZero()) {

    DivStep62(&u,&v,&eta,&pos,&uu,&uv,&vu,&vv);

    // Now update BigInt variables

    MatrixVecMul(&u,&v,uu,uv,vu,vv);

#if 1
    // Make u,v positive
    // Required only for Pornin's method
    if(u.IsNegative()) {
      u.Neg();
      uu = -uu;
      uv = -uv;
    }
    if(v.IsNegative()) {
      v.Neg();
      vu = -vu;
      vv = -vv;
    }
#endif

    MatrixVecMul(&r,&s,uu,uv,vu,vv,&carryR,&carryS);

    // Compute multiple of P to add to s and r to make them multiple of 2^62
    uint64_t r0 = (r.bits64[0] * MM64) & MSK62;
    uint64_t s0 = (s.bits64[0] * MM64) & MSK62;
    r0_P.Mult(&_P,r0);
    s0_P.Mult(&_P,s0);
    carryR = r.AddCh(&r0_P,carryR);
    carryS = s.AddCh(&s0_P,carryS);

    // Right shift all variables by 62bits
    shiftR(62, u.bits64);
    shiftR(62, v.bits64);
    shiftR(62, r.bits64, carryR);
    shiftR(62, s.bits64, carryS);

    //printf("U=%s\n",u.GetBase16().c_str());
    //printf("V=%s\n",v.GetBase16().c_str());
    //printf("R=%s\n",r.GetBase16().c_str());
    //printf("S=%s\n",s.GetBase16().c_str());
    totalCount++;

  }
  
  // u ends with +/-1
  if(u.IsNegative()) {
    u.Neg();
    r.Neg();
  }

  if (!u.IsOne()) {
    // No inverse
    CLEAR();
    return;
  }

  while(r.IsNegative())
    r.Add(&_P);
  while(r.IsGreaterOrEqual(&_P))
    r.Sub(&_P);

  Set(&r);

#endif

}

// ------------------------------------------------

void Int::ModExp(Int *e) {

  Int base(this);
  SetInt32(1);
  uint32_t i = 0;

  uint32_t nbBit = e->GetBitLength();
  for(int i=0;i<(int)nbBit;i++) {
    if (e->GetBit(i))
      ModMul(&base);
    base.ModMul(&base);
  }

}

// ------------------------------------------------

void Int::ModMul(Int *a) {

  Int p;
  p.MontgomeryMult(a, this);
  MontgomeryMult(&_R2, &p);

}

// ------------------------------------------------

void Int::ModSquare(Int *a) {

  Int p;
  p.MontgomeryMult(a, a);
  MontgomeryMult(&_R2, &p);

}

// ------------------------------------------------

void Int::ModCube(Int *a) {

  Int p;
  Int p2;
  p.MontgomeryMult(a, a);
  p2.MontgomeryMult(&p, a);
  MontgomeryMult(&_R3, &p2);

}

// ------------------------------------------------

bool Int::HasSqrt() {

  // Euler's criterion
  Int e(&_P);
  Int a(this);
  e.SubOne();
  e.ShiftR(1);
  a.ModExp(&e);

  return a.IsOne();

}

// ------------------------------------------------

void Int::ModSqrt() {

  if (_P.IsEven()) {
    CLEAR();
    return;
  }

  if (!HasSqrt()) {
    CLEAR();
    return;
  }

  if ((_P.bits64[0] & 3) == 3) {

    Int e(&_P);
    e.AddOne();
    e.ShiftR(2);
    ModExp(&e);

  } else if ((_P.bits64[0] & 3) == 1) {

    int nbBit = _P.GetBitLength();

    // Tonelli Shanks
    uint64_t e=0;
    Int S(&_P);
    S.SubOne();
    while (S.IsEven()) {
      S.ShiftR(1);
      e++;
    }

    // Search smalest non-qresidue of P
    Int q((uint64_t)1);
    do {
      q.AddOne();
    }  while (q.HasSqrt());

    Int c(&q);
    c.ModExp(&S);

    Int t(this);
    t.ModExp(&S);

    Int r(this);
    Int ex(&S);
    ex.AddOne();
    ex.ShiftR(1);
    r.ModExp(&ex);

    uint64_t M = e;
    while (!t.IsOne()) {

      Int t2(&t);
      uint64_t i=0;
      while (!t2.IsOne()) {
        t2.ModSquare(&t2);
        i++;
      }

      Int b(&c);
      for(uint64_t j=0;j<M-i-1;j++)
        b.ModSquare(&b);
      M=i;
      c.ModSquare(&b);
      t.ModMul(&t,&c);
      r.ModMul(&r,&b);

    }

    Set(&r);

  }

}

// ------------------------------------------------

void Int::ModMul(Int *a, Int *b) {

  Int p;
  p.MontgomeryMult(a,b);
  MontgomeryMult(&_R2,&p);

}

// ------------------------------------------------

Int* Int::GetFieldCharacteristic() {
  return &_P;
}

// ------------------------------------------------

Int* Int::GetR() {
  return &_R;
}
Int* Int::GetR2() {
  return &_R2;
}
Int* Int::GetR3() {
  return &_R3;
}
Int* Int::GetR4() {
  return &_R4;
}

// ------------------------------------------------

void Int::SetupField(Int *n, Int *R, Int *R2, Int *R3, Int *R4) {

  // Size in number of 32bit word
  int nSize = n->GetSize();

  // Last digit inversions (Newton's iteration)
  {
    int64_t x, t;
    x = t = (int64_t)n->bits64[0];
    x = x * (2 - t * x);
    x = x * (2 - t * x);
    x = x * (2 - t * x);
    x = x * (2 - t * x);
    x = x * (2 - t * x);
    MM64 = (uint64_t)(-x);
    MM32 = (uint32_t)MM64;
  }
  _P.Set(n);

  // Size of Montgomery mult (64bits digit)
  Msize = nSize/2;

  // Compute few power of R
  // R = 2^(64*Msize) mod n
  Int Ri;
  Ri.MontgomeryMult(&_ONE, &_ONE); // Ri = R^-1
  _R.Set(&Ri);                     // R  = R^-1
  _R2.MontgomeryMult(&Ri, &_ONE);  // R2 = R^-2
  _R3.MontgomeryMult(&Ri, &Ri);    // R3 = R^-3
  _R4.MontgomeryMult(&_R3, &_ONE); // R4 = R^-4

  _R.ModInv();                     // R  = R
  _R2.ModInv();                    // R2 = R^2
  _R3.ModInv();                    // R3 = R^3
  _R4.ModInv();                    // R4 = R^4

  if (R)
    R->Set(&_R);

  if (R2)
    R2->Set(&_R2);

  if (R3)
    R3->Set(&_R3);

  if (R4)
    R4->Set(&_R4);

}

// ------------------------------------------------
void Int::MontgomeryMult(Int *a) {

  // Compute a*b*R^-1 (mod n),  R=2^k (mod n), k = Msize*64
  // a and b must be lower than n
  // See SetupField()

  Int t;
  Int pr;
  Int p;
  uint64_t ML;
  uint64_t c;

  // i = 0
  imm_umul(a->bits64, bits64[0], pr.bits64);
  ML = pr.bits64[0] * MM64;
  imm_umul(_P.bits64, ML, p.bits64);
  c = pr.AddC(&p);
  memcpy(t.bits64, pr.bits64 + 1, 8 * (NB64BLOCK - 1));
  t.bits64[NB64BLOCK - 1] = c;

  for (int i = 1; i < Msize; i++) {

    imm_umul(a->bits64, bits64[i], pr.bits64);
    ML = (pr.bits64[0] + t.bits64[0]) * MM64;
    imm_umul(_P.bits64, ML, p.bits64);
	  c = pr.AddC(&p);
    t.AddAndShift(&t, &pr, c);

  }

  p.Sub(&t,&_P);
  if (p.IsPositive())
    Set(&p);
  else
    Set(&t);

}

void Int::MontgomeryMult(Int *a, Int *b) {

  // Compute a*b*R^-1 (mod n),  R=2^k (mod n), k = Msize*64
  // a and b must be lower than n
  // See SetupField()

  Int pr;
  Int p;
  uint64_t ML;
  uint64_t c;

  // i = 0
  imm_umul(a->bits64, b->bits64[0], pr.bits64);
  ML = pr.bits64[0] * MM64;
  imm_umul(_P.bits64, ML, p.bits64);
  c = pr.AddC(&p);
  memcpy(bits64,pr.bits64 + 1,8*(NB64BLOCK-1));
  bits64[NB64BLOCK-1] = c;

  for (int i = 1; i < Msize; i++) {

    imm_umul(a->bits64, b->bits64[i], pr.bits64);
    ML = (pr.bits64[0] + bits64[0]) * MM64;
    imm_umul(_P.bits64, ML, p.bits64);
	  c = pr.AddC(&p);
    AddAndShift(this, &pr, c);

  }

  p.Sub(this, &_P);
  if (p.IsPositive())
    Set(&p);

}


// SecpK1 specific section -----------------------------------------------------------------------------

void Int::ModMulK1(Int *a, Int *b) {

#ifndef WIN64
#if (__GNUC__ > 7) || (__GNUC__ == 7 && (__GNUC_MINOR__ > 2))
  unsigned char c;
#else
  #warning "GCC lass than 7.3 detected, upgrade gcc to get best perfromance"
  volatile unsigned char c;
#endif
#else
  unsigned char c;
#endif


  uint64_t ah, al;
  uint64_t t[NB64BLOCK];
#if BISIZE==256
  uint64_t r512[8];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
#else
  uint64_t r512[12];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
  r512[8] = 0;
  r512[9] = 0;
  r512[10] = 0;
  r512[11] = 0;
#endif

  // 256*256 multiplier
  imm_umul(a->bits64, b->bits64[0], r512);
  imm_umul(a->bits64, b->bits64[1], t);
  c = _addcarry_u64(0, r512[1], t[0], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[1], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[2], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[3], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[4], r512 + 5);
  imm_umul(a->bits64, b->bits64[2], t);
  c = _addcarry_u64(0, r512[2], t[0], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[1], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[2], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[3], r512 + 5);
  c = _addcarry_u64(c, r512[6], t[4], r512 + 6);
  imm_umul(a->bits64, b->bits64[3], t);
  c = _addcarry_u64(0, r512[3], t[0], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[1], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[2], r512 + 5);
  c = _addcarry_u64(c, r512[6], t[3], r512 + 6);
  c = _addcarry_u64(c, r512[7], t[4], r512 + 7);

  // Reduce from 512 to 320 
  imm_umul(r512 + 4, 0x1000003D1ULL, t);
  c = _addcarry_u64(0, r512[0], t[0], r512 + 0);
  c = _addcarry_u64(c, r512[1], t[1], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[2], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[3], r512 + 3);

  // Reduce from 320 to 256 
  // No overflow possible here t[4]+c<=0x1000003D1ULL
  al = _umul128(t[4] + c, 0x1000003D1ULL, &ah); 
  c = _addcarry_u64(0, r512[0], al, bits64 + 0);
  c = _addcarry_u64(c, r512[1], ah, bits64 + 1);
  c = _addcarry_u64(c, r512[2], 0ULL, bits64 + 2);
  c = _addcarry_u64(c, r512[3], 0ULL, bits64 + 3);

  // Probability of carry here or that this>P is very very unlikely
  bits64[4] = 0; 
#if BISIZE==512
  bits64[5] = 0;
  bits64[6] = 0;
  bits64[7] = 0;
  bits64[8] = 0;
#endif

}

void Int::ModMulK1(Int *a) {

#ifndef WIN64
#if (__GNUC__ > 7) || (__GNUC__ == 7 && (__GNUC_MINOR__ > 2))
  unsigned char c;
#else
  #warning "GCC lass than 7.3 detected, upgrade gcc to get best perfromance"
  volatile unsigned char c;
#endif
#else
  unsigned char c;
#endif

  uint64_t ah, al;
  uint64_t t[NB64BLOCK];
#if BISIZE==256
  uint64_t r512[8];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
#else
  uint64_t r512[12];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
  r512[8] = 0;
  r512[9] = 0;
  r512[10] = 0;
  r512[11] = 0;
#endif


  // 256*256 multiplier
  imm_umul(a->bits64, bits64[0], r512);
  imm_umul(a->bits64, bits64[1], t);
  c = _addcarry_u64(0, r512[1], t[0], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[1], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[2], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[3], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[4], r512 + 5);
  imm_umul(a->bits64, bits64[2], t);
  c = _addcarry_u64(0, r512[2], t[0], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[1], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[2], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[3], r512 + 5);
  c = _addcarry_u64(c, r512[6], t[4], r512 + 6);
  imm_umul(a->bits64, bits64[3], t);
  c = _addcarry_u64(0, r512[3], t[0], r512 + 3);
  c = _addcarry_u64(c, r512[4], t[1], r512 + 4);
  c = _addcarry_u64(c, r512[5], t[2], r512 + 5);
  c = _addcarry_u64(c, r512[6], t[3], r512 + 6);
  c = _addcarry_u64(c, r512[7], t[4], r512 + 7);

  // Reduce from 512 to 320 
  imm_umul(r512 + 4, 0x1000003D1ULL, t);
  c = _addcarry_u64(0, r512[0], t[0], r512 + 0);
  c = _addcarry_u64(c, r512[1], t[1], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[2], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[3], r512 + 3);

  // Reduce from 320 to 256 
  // No overflow possible here t[4]+c<=0x1000003D1ULL
  al = _umul128(t[4] + c, 0x1000003D1ULL, &ah);
  c = _addcarry_u64(0, r512[0], al, bits64 + 0);
  c = _addcarry_u64(c, r512[1], ah, bits64 + 1);
  c = _addcarry_u64(c, r512[2], 0, bits64 + 2);
  c = _addcarry_u64(c, r512[3], 0, bits64 + 3);
  // Probability of carry here or that this>P is very very unlikely
  bits64[4] = 0;
#if BISIZE==512
  bits64[5] = 0;
  bits64[6] = 0;
  bits64[7] = 0;
  bits64[8] = 0;
#endif

}

void Int::ModSquareK1(Int *a) {

#ifndef WIN64
#if (__GNUC__ > 7) || (__GNUC__ == 7 && (__GNUC_MINOR__ > 2))
  unsigned char c;
#else
  #warning "GCC lass than 7.3 detected, upgrade gcc to get best perfromance"
  volatile unsigned char c;
#endif
#else
  unsigned char c;
#endif

  uint64_t u10, u11;
  uint64_t t1;
  uint64_t t2;
  uint64_t t[NB64BLOCK];
#if BISIZE==256
  uint64_t r512[8];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
#else
  uint64_t r512[12];
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;
  r512[8] = 0;
  r512[9] = 0;
  r512[10] = 0;
  r512[11] = 0;
#endif


  //k=0
  r512[0] = _umul128(a->bits64[0], a->bits64[0], &t[1]);

  //k=1
  t[3] = _umul128(a->bits64[0], a->bits64[1], &t[4]);
  c = _addcarry_u64(0, t[3], t[3], &t[3]);
  c = _addcarry_u64(c, t[4], t[4], &t[4]);
  c = _addcarry_u64(c,  0,  0, &t1);
  c = _addcarry_u64(0, t[1], t[3], &t[3]);
  c = _addcarry_u64(c, t[4],  0, &t[4]);
  c = _addcarry_u64(c, t1,  0, &t1);
  r512[1] = t[3];

  //k=2
  t[0] = _umul128(a->bits64[0], a->bits64[2], &t[1]);
  c = _addcarry_u64(0, t[0], t[0], &t[0]);
  c = _addcarry_u64(c, t[1], t[1], &t[1]);
  c = _addcarry_u64(c,  0,  0, &t2);

  u10 = _umul128(a->bits64[1], a->bits64[1], &u11);
  c = _addcarry_u64(0, t[0] , u10, &t[0]);
  c = _addcarry_u64(c, t[1] , u11, &t[1]);
  c = _addcarry_u64(c, t2 ,   0, &t2);
  c = _addcarry_u64(0, t[0], t[4], &t[0]);
  c = _addcarry_u64(c, t[1], t1, &t[1]);
  c = _addcarry_u64(c, t2, 0, &t2);
  r512[2] = t[0];

  //k=3
  t[3] = _umul128(a->bits64[0], a->bits64[3], &t[4]);
  u10 = _umul128(a->bits64[1], a->bits64[2], &u11);

  c = _addcarry_u64(0, t[3], u10, &t[3]);
  c = _addcarry_u64(c, t[4], u11, &t[4]);
  c = _addcarry_u64(c,  0,   0, &t1);
  t1 += t1;
  c = _addcarry_u64(0, t[3], t[3], &t[3]);
  c = _addcarry_u64(c, t[4], t[4], &t[4]);
  c = _addcarry_u64(c, t1, 0, &t1);
  c = _addcarry_u64(0, t[3], t[1], &t[3]);
  c = _addcarry_u64(c, t[4], t2, &t[4]);
  c = _addcarry_u64(c, t1, 0, &t1);
  r512[3] = t[3];

  //k=4
  t[0] = _umul128(a->bits64[1], a->bits64[3], &t[1]);
  c = _addcarry_u64(0, t[0], t[0], &t[0]);
  c = _addcarry_u64(c, t[1], t[1], &t[1]);
  c = _addcarry_u64(c, 0, 0, &t2);

  u10 = _umul128(a->bits64[2], a->bits64[2], &u11);
  c = _addcarry_u64(0, t[0], u10, &t[0]);
  c = _addcarry_u64(c, t[1], u11, &t[1]);
  c = _addcarry_u64(c, t2, 0, &t2);
  c = _addcarry_u64(0, t[0], t[4], &t[0]);
  c = _addcarry_u64(c, t[1], t1, &t[1]);
  c = _addcarry_u64(c, t2,  0, &t2);
  r512[4] = t[0];

  //k=5
  t[3] = _umul128(a->bits64[2], a->bits64[3], &t[4]);
  c = _addcarry_u64(0, t[3], t[3], &t[3]);
  c = _addcarry_u64(c, t[4], t[4], &t[4]);
  c = _addcarry_u64(c, 0, 0, &t1);
  c = _addcarry_u64(0, t[3], t[1], &t[3]);
  c = _addcarry_u64(c, t[4], t2, &t[4]);
  c = _addcarry_u64(c, t1,  0, &t1);
  r512[5] = t[3];

  //k=6
  t[0] = _umul128(a->bits64[3], a->bits64[3], &t[1]);
  c = _addcarry_u64(0, t[0], t[4], &t[0]);
  c = _addcarry_u64(c, t[1], t1, &t[1]);
  r512[6] = t[0];

  //k=7
  r512[7] = t[1];

  // Reduce from 512 to 320 
  // Reduce from 512 to 320 
  imm_umul(r512 + 4, 0x1000003D1ULL, t);
  c = _addcarry_u64(0, r512[0], t[0], r512 + 0);
  c = _addcarry_u64(c, r512[1], t[1], r512 + 1);
  c = _addcarry_u64(c, r512[2], t[2], r512 + 2);
  c = _addcarry_u64(c, r512[3], t[3], r512 + 3);

  // Reduce from 320 to 256 
  // No overflow possible here t[4]+c<=0x1000003D1ULL
  u10 = _umul128(t[4] + c, 0x1000003D1ULL, &u11);
  c = _addcarry_u64(0, r512[0], u10, bits64 + 0);
  c = _addcarry_u64(c, r512[1], u11, bits64 + 1);
  c = _addcarry_u64(c, r512[2], 0, bits64 + 2);
  c = _addcarry_u64(c, r512[3], 0, bits64 + 3);
  // Probability of carry here or that this>P is very very unlikely
  bits64[4] = 0;
#if BISIZE==512
  bits64[5] = 0;
  bits64[6] = 0;
  bits64[7] = 0;
  bits64[8] = 0;
#endif

}

static Int _R2o;                               // R^2 for SecpK1 order modular mult
static uint64_t MM64o = 0x4B0DFF665588B13FULL; // 64bits lsb negative inverse of SecpK1 order
static Int *_O;                                // SecpK1 order

void Int::InitK1(Int *order) {
  _O = order;
  _R2o.SetBase16("9D671CD581C69BC5E697F5E45BCD07C6741496C20E7CF878896CF21467D7D140");
}

void Int::ModAddK1order(Int *a, Int *b) {
  Add(a,b);
  Sub(_O);
  if (IsNegative())
    Add(_O);
}

void Int::ModAddK1order(Int *a) {
  Add(a);
  Sub(_O);
  if(IsNegative())
    Add(_O);
}

void Int::ModSubK1order(Int *a) {
  Sub(a);
  if(IsNegative())
    Add(_O);
}

void Int::ModNegK1order() {
  Neg();
  Add(_O);
}

uint32_t Int::ModPositiveK1() {
  
  Int N(this);
  Int D(this);
  N.ModNeg();
  D.Sub(&N);
  if(D.IsNegative()) {
    return 0;
  } else {
    Set(&N);
    return 1;
  }

}


void Int::ModMulK1order(Int *a) {

  Int t;
  Int pr;
  Int p;
  uint64_t ML;
  uint64_t c;

  imm_umul(a->bits64, bits64[0], pr.bits64);
  ML = pr.bits64[0] * MM64o;
  imm_umul(_O->bits64, ML, p.bits64);
  c = pr.AddC(&p);
  memcpy(t.bits64,pr.bits64 + 1,8 * (NB64BLOCK - 1));
  t.bits64[NB64BLOCK - 1] = c;

  for (int i = 1; i < 4; i++) {

    imm_umul(a->bits64, bits64[i], pr.bits64);
    ML = (pr.bits64[0] + t.bits64[0]) * MM64o;
    imm_umul(_O->bits64, ML, p.bits64);
    c = pr.AddC(&p);
    t.AddAndShift(&t, &pr, c);

  }

  p.Sub(&t, _O);
  if (p.IsPositive())
    Set(&p);
  else
    Set(&t);

  // Normalize

  imm_umul(_R2o.bits64, bits64[0], pr.bits64);
  ML = pr.bits64[0] * MM64o;
  imm_umul(_O->bits64, ML, p.bits64);
  c = pr.AddC(&p);
  memcpy(t.bits64,pr.bits64 + 1,8 * (NB64BLOCK - 1));
  t.bits64[NB64BLOCK - 1] = c;

  for (int i = 1; i < 4; i++) {

    imm_umul(_R2o.bits64, bits64[i], pr.bits64);
    ML = (pr.bits64[0] + t.bits64[0]) * MM64o;
    imm_umul(_O->bits64, ML, p.bits64);
    c = pr.AddC(&p);
    t.AddAndShift(&t, &pr, c);

  }

  p.Sub(&t, _O);
  if (p.IsPositive())
    Set(&p);
  else
    Set(&t);

}
