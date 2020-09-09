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
#include "IntGroup.h"
#include <string.h>
#include <math.h>
#include <emmintrin.h>
#include "Timer.h"

#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

Int _ONE((uint64_t)1);


// ------------------------------------------------

Int::Int() {
}

Int::Int(Int *a) {
  if(a) Set(a);
  else CLEAR();
}

Int::Int(int64_t i64) {

  if (i64 < 0) {
	  CLEARFF();
  } else {
	  CLEAR();
  }
  bits64[0] = i64;

}

Int::Int(uint64_t u64) {

  CLEAR();
  bits64[0] = u64;

}

// ------------------------------------------------

void Int::CLEAR() {

  memset(bits64,0, NB64BLOCK*8);

}

void Int::CLEARFF() {

  memset(bits64, 0xFF, NB64BLOCK * 8);

}

// ------------------------------------------------

void Int::Set(Int *a) {

  for (int i = 0; i<NB64BLOCK; i++)
  	bits64[i] = a->bits64[i];

}

// ------------------------------------------------

void Int::Add(Int *a) {

  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0], a->bits64[0], bits64 +0);
  c = _addcarry_u64(c, bits64[1], a->bits64[1], bits64 +1);
  c = _addcarry_u64(c, bits64[2], a->bits64[2], bits64 +2);
  c = _addcarry_u64(c, bits64[3], a->bits64[3], bits64 +3);
  c = _addcarry_u64(c, bits64[4], a->bits64[4], bits64 +4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, bits64[5], a->bits64[5], bits64 +5);
  c = _addcarry_u64(c, bits64[6], a->bits64[6], bits64 +6);
  c = _addcarry_u64(c, bits64[7], a->bits64[7], bits64 +7);
  c = _addcarry_u64(c, bits64[8], a->bits64[8], bits64 +8);
#endif

}

// ------------------------------------------------

void Int::Add(uint64_t a) {

	unsigned char c = 0;
	c = _addcarry_u64(c, bits64[0], a, bits64 + 0);
	c = _addcarry_u64(c, bits64[1], 0, bits64 + 1);
	c = _addcarry_u64(c, bits64[2], 0, bits64 + 2);
	c = _addcarry_u64(c, bits64[3], 0, bits64 + 3);
	c = _addcarry_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
	c = _addcarry_u64(c, bits64[5], 0, bits64 + 5);
	c = _addcarry_u64(c, bits64[6], 0, bits64 + 6);
	c = _addcarry_u64(c, bits64[7], 0, bits64 + 7);
	c = _addcarry_u64(c, bits64[8], 0, bits64 + 8);
#endif
}

// ------------------------------------------------
void Int::AddOne() {

  unsigned char c = 0;
  c = _addcarry_u64(c, bits64[0],1, bits64 +0);
  c = _addcarry_u64(c, bits64[1],0, bits64 +1);
  c = _addcarry_u64(c, bits64[2],0, bits64 +2);
  c = _addcarry_u64(c, bits64[3],0, bits64 +3);
  c = _addcarry_u64(c, bits64[4],0, bits64 +4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, bits64[5],0, bits64 +5);
  c = _addcarry_u64(c, bits64[6],0, bits64 +6);
  c = _addcarry_u64(c, bits64[7],0, bits64 +7);
  c = _addcarry_u64(c, bits64[8],0, bits64 +8);
#endif

}

// ------------------------------------------------

void Int::Add(Int *a,Int *b) {

  unsigned char c = 0;
  c = _addcarry_u64(c, b->bits64[0], a->bits64[0], bits64 +0);
  c = _addcarry_u64(c, b->bits64[1], a->bits64[1], bits64 +1);
  c = _addcarry_u64(c, b->bits64[2], a->bits64[2], bits64 +2);
  c = _addcarry_u64(c, b->bits64[3], a->bits64[3], bits64 +3);
  c = _addcarry_u64(c, b->bits64[4], a->bits64[4], bits64 +4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c, b->bits64[5], a->bits64[5], bits64 +5);
  c = _addcarry_u64(c, b->bits64[6], a->bits64[6], bits64 +6);
  c = _addcarry_u64(c, b->bits64[7], a->bits64[7], bits64 +7);
  c = _addcarry_u64(c, b->bits64[8], a->bits64[8], bits64 +8);
#endif

}

// ------------------------------------------------

uint64_t Int::AddCh(Int* a,uint64_t ca,Int* b,uint64_t cb) {

  uint64_t carry;
  unsigned char c = 0;
  c = _addcarry_u64(c,a->bits64[0],b->bits64[0],bits64 + 0);
  c = _addcarry_u64(c,a->bits64[1],b->bits64[1],bits64 + 1);
  c = _addcarry_u64(c,a->bits64[2],b->bits64[2],bits64 + 2);
  c = _addcarry_u64(c,a->bits64[3],b->bits64[3],bits64 + 3);
  c = _addcarry_u64(c,a->bits64[4],b->bits64[4],bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c,a->bits64[5],b->bits64[5],bits64 + 5);
  c = _addcarry_u64(c,a->bits64[6],b->bits64[6],bits64 + 6);
  c = _addcarry_u64(c,a->bits64[7],b->bits64[7],bits64 + 7);
  c = _addcarry_u64(c,a->bits64[8],b->bits64[8],bits64 + 8);
#endif
  _addcarry_u64(c,ca,cb,&carry);
  return carry;

}

uint64_t Int::AddCh(Int* a,uint64_t ca) {

  uint64_t carry;
  unsigned char c = 0;
  c = _addcarry_u64(c,bits64[0],a->bits64[0],bits64 + 0);
  c = _addcarry_u64(c,bits64[1],a->bits64[1],bits64 + 1);
  c = _addcarry_u64(c,bits64[2],a->bits64[2],bits64 + 2);
  c = _addcarry_u64(c,bits64[3],a->bits64[3],bits64 + 3);
  c = _addcarry_u64(c,bits64[4],a->bits64[4],bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c,bits64[5],a->bits64[5],bits64 + 5);
  c = _addcarry_u64(c,bits64[6],a->bits64[6],bits64 + 6);
  c = _addcarry_u64(c,bits64[7],a->bits64[7],bits64 + 7);
  c = _addcarry_u64(c,bits64[8],a->bits64[8],bits64 + 8);
#endif
  _addcarry_u64(c,ca,0,&carry);
  return carry;

}
// ------------------------------------------------

uint64_t Int::AddC(Int* a) {

  unsigned char c = 0;
  c = _addcarry_u64(c,bits64[0],a->bits64[0],bits64 + 0);
  c = _addcarry_u64(c,bits64[1],a->bits64[1],bits64 + 1);
  c = _addcarry_u64(c,bits64[2],a->bits64[2],bits64 + 2);
  c = _addcarry_u64(c,bits64[3],a->bits64[3],bits64 + 3);
  c = _addcarry_u64(c,bits64[4],a->bits64[4],bits64 + 4);
#if NB64BLOCK > 5
  c = _addcarry_u64(c,bits64[5],a->bits64[5],bits64 + 5);
  c = _addcarry_u64(c,bits64[6],a->bits64[6],bits64 + 6);
  c = _addcarry_u64(c,bits64[7],a->bits64[7],bits64 + 7);
  c = _addcarry_u64(c,bits64[8],a->bits64[8],bits64 + 8);
#endif

  return c;

}

// ------------------------------------------------

void Int::AddAndShift(Int* a,Int* b,uint64_t cH) {

  unsigned char c = 0;
  c = _addcarry_u64(c,b->bits64[0],a->bits64[0],bits64 + 0);
  c = _addcarry_u64(c,b->bits64[1],a->bits64[1],bits64 + 0);
  c = _addcarry_u64(c,b->bits64[2],a->bits64[2],bits64 + 1);
  c = _addcarry_u64(c,b->bits64[3],a->bits64[3],bits64 + 2);
  c = _addcarry_u64(c,b->bits64[4],a->bits64[4],bits64 + 3);
#if NB64BLOCK > 5
  c = _addcarry_u64(c,b->bits64[5],a->bits64[5],bits64 + 4);
  c = _addcarry_u64(c,b->bits64[6],a->bits64[6],bits64 + 5);
  c = _addcarry_u64(c,b->bits64[7],a->bits64[7],bits64 + 6);
  c = _addcarry_u64(c,b->bits64[8],a->bits64[8],bits64 + 7);
#endif

  bits64[NB64BLOCK - 1] = c + cH;

}

// ------------------------------------------------

void Int::MatrixVecMul(Int* u,Int* v,int64_t _11,int64_t _12,int64_t _21,int64_t _22,uint64_t* cu,uint64_t* cv) {

  Int t1,t2,t3,t4;
  uint64_t c1,c2,c3,c4;
  c1 = t1.IMult(u,_11);
  c2 = t2.IMult(v,_12);
  c3 = t3.IMult(u,_21);
  c4 = t4.IMult(v,_22);
  *cu = u->AddCh(&t1,c1,&t2,c2);
  *cv = v->AddCh(&t3,c3,&t4,c4);

}

void Int::MatrixVecMul(Int* u,Int* v,int64_t _11,int64_t _12,int64_t _21,int64_t _22) {

  Int t1,t2,t3,t4;
  t1.IMult(u,_11);
  t2.IMult(v,_12);
  t3.IMult(u,_21);
  t4.IMult(v,_22);
  u->Add(&t1,&t2);
  v->Add(&t3,&t4);

}

/*
void Int::MatrixVecMul(Int* u,Int* v,int64_t _11,int64_t _12,int64_t _21,int64_t _22,int len,bool *negu,bool *negv) {

  Int t1,t2,t3,t4;
  Int* du1;
  Int* du2;
  Int* dv1;
  Int* dv2;
  Int nu;
  Int nv;
  unsigned char c1;
  unsigned char c2;
  unsigned char c3;
  unsigned char c4;
  uint64_t h1,carry1;
  uint64_t h2,carry2;
  uint64_t h3,carry3;
  uint64_t h4,carry4;

  // Compute -u,-v
  c1 = _subborrow_u64(0,0,u->bits64[0],nu.bits64 + 0);
  c2 = _subborrow_u64(0,0,v->bits64[0],nv.bits64 + 0);
  for(int i = 1; i <= len; i++) {
    c1 = _subborrow_u64(c1,0,u->bits64[i],nu.bits64 + i);
    c2 = _subborrow_u64(c2,0,v->bits64[i],nv.bits64 + i);
  }

  // Make _XY positive
  if(_11 < 0) {
    du1 = &nu;
    _11 = -_11;
  } else {
    du1 = u;
  }
  if(_12 < 0) {
    dv1 = &nv;
    _12 = -_12;
  } else {
    dv1 = v;
  }
  if(_21 < 0) {
    du2 = &nu;
    _21 = -_21;
  } else {
    du2 = u;
  }
  if(_22 < 0) {
    dv2 = &nv;
    _22 = -_22;
  } else {
    dv2 = v;
  }

  // Compute product
  t1.bits64[0] = _umul128(du1->bits64[0],_11,&h1); carry1 = h1;
  t2.bits64[0] = _umul128(dv1->bits64[0],_12,&h2); carry2 = h2;
  t3.bits64[0] = _umul128(du2->bits64[0],_21,&h3); carry3 = h3;
  t4.bits64[0] = _umul128(dv2->bits64[0],_22,&h4); carry4 = h4;
  c1 = 0; c2 = 0; c3 = 0; c4 = 0;

  for(int i = 1; i <= len; i++) {
    c1 = _addcarry_u64(c1,_umul128(du1->bits64[i],_11,&h1),carry1,t1.bits64 + i); carry1 = h1;
    c2 = _addcarry_u64(c2,_umul128(dv1->bits64[i],_12,&h2),carry2,t2.bits64 + i); carry2 = h2;
    c3 = _addcarry_u64(c3,_umul128(du2->bits64[i],_21,&h3),carry3,t3.bits64 + i); carry3 = h3;
    c4 = _addcarry_u64(c4,_umul128(dv2->bits64[i],_22,&h4),carry4,t4.bits64 + i); carry4 = h4;
  }

  // Add
  c1 = 0; c2 = 0;
  for(int i = 0; i <= len; i++) {
    c1 = _addcarry_u64(c1,t1.bits64[i],t2.bits64[i],u->bits64 + i);
    c2 = _addcarry_u64(c2,t3.bits64[i],t4.bits64[i],v->bits64 + i);
  }

  *negu = (int64_t)u->bits64[len] < 0;
  *negv = (int64_t)v->bits64[len] < 0;

  if( *negu ) {
    c1 = 0;
    for(int i = 0; i <= len; i++)
      c1 = _subborrow_u64(c1,0,u->bits64[i],u->bits64 + i);
  }

  if( *negv ) {
    c1 = 0;
    for(int i = 0; i <= len; i++)
      c1 = _subborrow_u64(c1,0,v->bits64[i],v->bits64 + i);
  }


}
*/

// ------------------------------------------------

bool Int::IsGreater(Int *a) {

  int i;

  for(i=NB64BLOCK-1;i>=0;) {
    if( a->bits64[i]!= bits64[i] )
		break;
    i--;
  }

  if(i>=0) {
    return bits64[i]>a->bits64[i];
  } else {
    return false;
  }

}

// ------------------------------------------------

bool Int::IsLower(Int *a) {

  int i;

  for (i = NB64BLOCK - 1; i >= 0;) {
    if (a->bits64[i] != bits64[i])
      break;
    i--;
  }

  if (i >= 0) {
    return bits64[i]<a->bits64[i];
  } else {
    return false;
  }

}

// ------------------------------------------------

bool Int::IsGreaterOrEqual(Int *a) {

  Int p;
  p.Sub(this,a);
  return p.IsPositive();

}

// ------------------------------------------------

bool Int::IsLowerOrEqual(Int *a) {

  int i = NB64BLOCK - 1;

  while (i >= 0) {
    if (a->bits64[i] != bits64[i])
      break;
    i--;
}

  if (i >= 0) {
    return bits64[i]<a->bits64[i];
  } else {
    return true;
  }

}

bool Int::IsEqual(Int *a) {

return

#if NB64BLOCK > 5
  (bits64[8] == a->bits64[8]) &&
  (bits64[7] == a->bits64[7]) &&
  (bits64[6] == a->bits64[6]) &&
  (bits64[5] == a->bits64[5]) &&
#endif
  (bits64[4] == a->bits64[4]) &&
  (bits64[3] == a->bits64[3]) &&
  (bits64[2] == a->bits64[2]) &&
  (bits64[1] == a->bits64[1]) &&
  (bits64[0] == a->bits64[0]);

}

bool Int::IsOne() {
  return IsEqual(&_ONE);
}

bool Int::IsZero() {

#if NB64BLOCK > 5
  return (bits64[8] | bits64[7] | bits64[6] | bits64[5] | bits64[4] | bits64[3] | bits64[2] | bits64[1] | bits64[0]) == 0;
#else
  return (bits64[4] | bits64[3] | bits64[2] | bits64[1] | bits64[0]) == 0;
#endif

}


// ------------------------------------------------

void Int::SetInt32(uint32_t value) {

  CLEAR();
  bits[0]=value;

}

// ------------------------------------------------

uint32_t Int::GetInt32() {
  return bits[0];
}

// ------------------------------------------------

unsigned char Int::GetByte(int n) {
  
  unsigned char *bbPtr = (unsigned char *)bits;
  return bbPtr[n];

}

void Int::Set32Bytes(unsigned char *bytes) {

  CLEAR();
  uint64_t *ptr = (uint64_t *)bytes;
  bits64[3] = _byteswap_uint64(ptr[0]);
  bits64[2] = _byteswap_uint64(ptr[1]);
  bits64[1] = _byteswap_uint64(ptr[2]);
  bits64[0] = _byteswap_uint64(ptr[3]);

}

void Int::Get32Bytes(unsigned char *buff) {

  uint64_t *ptr = (uint64_t *)buff;
  ptr[3] = _byteswap_uint64(bits64[0]);
  ptr[2] = _byteswap_uint64(bits64[1]);
  ptr[1] = _byteswap_uint64(bits64[2]);
  ptr[0] = _byteswap_uint64(bits64[3]);

}

// ------------------------------------------------

void Int::SetByte(int n,unsigned char byte) {

	unsigned char *bbPtr = (unsigned char *)bits;
	bbPtr[n] = byte;

}

// ------------------------------------------------

void Int::SetDWord(int n,uint32_t b) {
  bits[n] = b;
}

// ------------------------------------------------

void Int::SetQWord(int n, uint64_t b) {
	bits64[n] = b;
}

// ------------------------------------------------

void Int::Sub(Int *a) {

  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], a->bits64[0], bits64 +0);
  c = _subborrow_u64(c, bits64[1], a->bits64[1], bits64 +1);
  c = _subborrow_u64(c, bits64[2], a->bits64[2], bits64 +2);
  c = _subborrow_u64(c, bits64[3], a->bits64[3], bits64 +3);
  c = _subborrow_u64(c, bits64[4], a->bits64[4], bits64 +4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, bits64[5], a->bits64[5], bits64 +5);
  c = _subborrow_u64(c, bits64[6], a->bits64[6], bits64 +6);
  c = _subborrow_u64(c, bits64[7], a->bits64[7], bits64 +7);
  c = _subborrow_u64(c, bits64[8], a->bits64[8], bits64 +8);
#endif

}

// ------------------------------------------------

void Int::Sub(Int *a,Int *b) {

  unsigned char c = 0;
  c = _subborrow_u64(c, a->bits64[0], b->bits64[0], bits64 + 0);
  c = _subborrow_u64(c, a->bits64[1], b->bits64[1], bits64 + 1);
  c = _subborrow_u64(c, a->bits64[2], b->bits64[2], bits64 + 2);
  c = _subborrow_u64(c, a->bits64[3], b->bits64[3], bits64 + 3);
  c = _subborrow_u64(c, a->bits64[4], b->bits64[4], bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, a->bits64[5], b->bits64[5], bits64 + 5);
  c = _subborrow_u64(c, a->bits64[6], b->bits64[6], bits64 + 6);
  c = _subborrow_u64(c, a->bits64[7], b->bits64[7], bits64 + 7);
  c = _subborrow_u64(c, a->bits64[8], b->bits64[8], bits64 + 8);
#endif

}

void Int::Sub(uint64_t a) {

  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], a, bits64 + 0);
  c = _subborrow_u64(c, bits64[1], 0, bits64 + 1);
  c = _subborrow_u64(c, bits64[2], 0, bits64 + 2);
  c = _subborrow_u64(c, bits64[3], 0, bits64 + 3);
  c = _subborrow_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, bits64[5], 0, bits64 + 5);
  c = _subborrow_u64(c, bits64[6], 0, bits64 + 6);
  c = _subborrow_u64(c, bits64[7], 0, bits64 + 7);
  c = _subborrow_u64(c, bits64[8], 0, bits64 + 8);
#endif

}

void Int::SubOne() {

  unsigned char c = 0;
  c = _subborrow_u64(c, bits64[0], 1, bits64 + 0);
  c = _subborrow_u64(c, bits64[1], 0, bits64 + 1);
  c = _subborrow_u64(c, bits64[2], 0, bits64 + 2);
  c = _subborrow_u64(c, bits64[3], 0, bits64 + 3);
  c = _subborrow_u64(c, bits64[4], 0, bits64 + 4);
#if NB64BLOCK > 5
  c = _subborrow_u64(c, bits64[5], 0, bits64 + 5);
  c = _subborrow_u64(c, bits64[6], 0, bits64 + 6);
  c = _subborrow_u64(c, bits64[7], 0, bits64 + 7);
  c = _subborrow_u64(c, bits64[8], 0, bits64 + 8);
#endif

}

// ------------------------------------------------

bool Int::IsPositive() {
  return (int64_t)(bits64[NB64BLOCK - 1])>=0;
}

// ------------------------------------------------

bool Int::IsNegative() {
  return (int64_t)(bits64[NB64BLOCK - 1])<0;
}

// ------------------------------------------------

bool Int::IsStrictPositive() {
  if( IsPositive() )
	  return !IsZero();
  else
	  return false;
}

// ------------------------------------------------

bool Int::IsEven() {
  return (bits[0] & 0x1) == 0;
}

// ------------------------------------------------

bool Int::IsOdd() {
  return (bits[0] & 0x1) == 1;
}

// ------------------------------------------------

void Int::Neg() {

	unsigned char c=0;
	c = _subborrow_u64(c, 0, bits64[0], bits64 + 0);
	c = _subborrow_u64(c, 0, bits64[1], bits64 + 1);
	c = _subborrow_u64(c, 0, bits64[2], bits64 + 2);
	c = _subborrow_u64(c, 0, bits64[3], bits64 + 3);
	c = _subborrow_u64(c, 0, bits64[4], bits64 + 4);
#if NB64BLOCK > 5
	c = _subborrow_u64(c, 0, bits64[5], bits64 + 5);
	c = _subborrow_u64(c, 0, bits64[6], bits64 + 6);
	c = _subborrow_u64(c, 0, bits64[7], bits64 + 7);
	c = _subborrow_u64(c, 0, bits64[8], bits64 + 8);
#endif

}

// ------------------------------------------------

void Int::ShiftL32Bit() {

  for(int i=NB32BLOCK-1;i>0;i--) {
    bits[i]=bits[i-1];
  }
  bits[0]=0;

}

// ------------------------------------------------

void Int::ShiftL64Bit() {

	for (int i = NB64BLOCK-1 ; i>0; i--) {
		bits64[i] = bits64[i - 1];
	}
	bits64[0] = 0;

}

// ------------------------------------------------

void Int::ShiftL64BitAndSub(Int *a,int n) {

  Int b;
  int i=NB64BLOCK-1;

  for(;i>=n;i--)
    b.bits64[i] = ~a->bits64[i-n];
  for(;i>=0;i--)
    b.bits64[i] = 0xFFFFFFFFFFFFFFFFULL;

  Add(&b);
  AddOne();

}

// ------------------------------------------------

void Int::ShiftL(uint32_t n) {

  if(n==0)
    return;
    
  if( n<64 ) {
	  shiftL((unsigned char)n, bits64);
  } else {
    uint32_t nb64 = n/64;
    uint32_t nb   = n%64;
    for(uint32_t i=0;i<nb64;i++) ShiftL64Bit();
	  shiftL((unsigned char)nb, bits64);
  }
  
}

// ------------------------------------------------

void Int::ShiftR32Bit() {

  for(int i=0;i<NB32BLOCK-1;i++) {
    bits[i]=bits[i+1];
  }
  if(((int32_t)bits[NB32BLOCK-2])<0)
    bits[NB32BLOCK-1] = 0xFFFFFFFF;
  else
    bits[NB32BLOCK-1]=0;

}

// ------------------------------------------------

void Int::ShiftR64Bit() {

	for (int i = 0; i<NB64BLOCK - 1; i++) {
		bits64[i] = bits64[i + 1];
	}
	if (((int64_t)bits64[NB64BLOCK - 2])<0)
		bits64[NB64BLOCK - 1] = 0xFFFFFFFFFFFFFFFF;
	else
		bits64[NB64BLOCK - 1] = 0;

}

// ---------------------------------D---------------

void Int::ShiftR(uint32_t n) {

  if(n==0)
    return;
    
  if( n<64 ) {
    shiftR((unsigned char)n, bits64);
  } else {
    uint32_t nb64 = n/64;
    uint32_t nb   = n%64;
    for(uint32_t i=0;i<nb64;i++) ShiftR64Bit();
	  shiftR((unsigned char)nb, bits64);
  }
  
}

// ------------------------------------------------

void Int::SwapBit(int bitNumber) {

  uint32_t nb64 = bitNumber / 64;
  uint32_t nb = bitNumber % 64;
  uint64_t mask = 1ULL << nb;
  if(bits64[nb64] & mask ) {
    bits64[nb64] &= ~mask;
  } else {
    bits64[nb64] |= mask;
  }

}

// ------------------------------------------------

void Int::Mult(Int *a) {

  Int b(this);
  Mult(a,&b);

}

// ------------------------------------------------

uint64_t Int::IMult(int64_t a) {

  uint64_t carry;

	// Make a positive
	if (a < 0LL) {
		a = -a;
		Neg();
	}

	imm_imul(bits64, a, bits64, &carry);
  return carry;

}

// ------------------------------------------------

uint64_t Int::Mult(uint64_t a) {

  uint64_t carry;
  imm_mul(bits64, a, bits64, &carry);
  return carry;

}
// ------------------------------------------------

uint64_t Int::IMult(Int *a, int64_t b) {
  
  uint64_t carry;

  // Make b positive
  if (b < 0LL) {

    unsigned char c = 0;
    c = _subborrow_u64(c,0,a->bits64[0],bits64 + 0);
    c = _subborrow_u64(c,0,a->bits64[1],bits64 + 1);
    c = _subborrow_u64(c,0,a->bits64[2],bits64 + 2);
    c = _subborrow_u64(c,0,a->bits64[3],bits64 + 3);
    c = _subborrow_u64(c,0,a->bits64[4],bits64 + 4);
#if NB64BLOCK > 5
    c = _subborrow_u64(c,0,a->bits64[5],bits64 + 5);
    c = _subborrow_u64(c,0,a->bits64[6],bits64 + 6);
    c = _subborrow_u64(c,0,a->bits64[7],bits64 + 7);
    c = _subborrow_u64(c,0,a->bits64[8],bits64 + 8);
#endif

  	b = -b;

  } else {

    Set(a);

  }

  imm_imul(bits64, b, bits64, &carry);
  return carry;

}


// ------------------------------------------------

uint64_t Int::Mult(Int *a, uint64_t b) {

  uint64_t carry;
  imm_mul(a->bits64, b, bits64, &carry);
  return carry;

}

// ------------------------------------------------

void Int::Mult(Int *a,Int *b) {
  
  unsigned char c = 0;
  uint64_t h;
  uint64_t pr = 0;
  uint64_t carryh = 0;
  uint64_t carryl = 0;

  bits64[0] = _umul128(a->bits64[0], b->bits64[0], &pr);

  for (int i = 1; i < NB64BLOCK; i++) {
    for (int j = 0; j <= i; j++) {
      c = _addcarry_u64(c, _umul128(a->bits64[j], b->bits64[i - j], &h), pr, &pr);
      c = _addcarry_u64(c, carryl, h, &carryl);
      c = _addcarry_u64(c, carryh, 0, &carryh);
    }
    bits64[i] = pr;
    pr = carryl;
    carryl = carryh;
    carryh = 0;
  }

}

// ------------------------------------------------

uint64_t Int::Mult(Int *a,uint32_t b) {
  uint64_t carry;
  imm_mul(a->bits64, (uint64_t)b, bits64, &carry);
  return carry;
}

// ------------------------------------------------

double Int::ToDouble() {

  double base = 1.0;
  double sum = 0;
  double pw32 = pow(2.0,32.0);
  for(int i=0;i<NB32BLOCK;i++) {
    sum += (double)(bits[i]) * base;
    base *= pw32;
  }

  return sum;

}

// ------------------------------------------------

int Int::GetBitLength() {

  Int t(this);
  if(IsNegative())
	  t.Neg();

  int i=NB64BLOCK-1;
  while(i>=0 && t.bits64[i]==0) i--;
  if(i<0) return 0;
  return (int)((64-LZC(t.bits64[i])) + i*64);

}

// ------------------------------------------------

int Int::GetSize() {

  int i=NB32BLOCK-1;
  while(i>0 && bits[i]==0) i--;
  return i+1;

}

// ------------------------------------------------

int Int::GetSize64() {

  int i = NB64BLOCK - 1;
  while(i > 0 && bits64[i] == 0) i--;
  return i + 1;

}

// ------------------------------------------------

void Int::MultModN(Int *a,Int *b,Int *n) {

  Int r;
  Mult(a,b);
  Div(n,&r);
  Set(&r);

}

// ------------------------------------------------

void Int::Mod(Int *n) {

  Int r;
  Div(n,&r);
  Set(&r);

}

// ------------------------------------------------

int Int::GetLowestBit() {

  // Assume this!=0
  int b=0;
  while(GetBit(b)==0) b++;
  return b;

}

// ------------------------------------------------

void Int::MaskByte(int n) {

  for (int i = n; i < NB32BLOCK; i++)
	  bits[i] = 0;

}

// ------------------------------------------------

void Int::Abs() {

  if (IsNegative())
    Neg();

}

// ------------------------------------------------

void Int::Rand(int nbit) {

	CLEAR();

	uint32_t nb = nbit/32;
	uint32_t leftBit = nbit%32;
	uint32_t mask = 1;
	mask = (mask << leftBit) - 1;
	uint32_t i=0;
	for(;i<nb;i++)
		bits[i]=rndl();
	bits[i]=rndl()&mask;

}

// ------------------------------------------------

void Int::Rand(Int *randMax) {

  int b = randMax->GetBitLength();
  Int r;
  r.Rand(b);
  Int q(&r);
  Int rem;
  q.Div(randMax,&rem);
  Set(&rem);

}

// ------------------------------------------------

void Int::Div(Int *a,Int *mod) {

  if(a->IsGreater(this)) {
    if(mod) mod->Set(this);
    CLEAR();
    return;
  }

  if(a->IsZero()) {
    printf("Divide by 0!\n");
    return;
  }

  if(IsEqual(a)) {
    if(mod) mod->CLEAR();
    Set(&_ONE);
    return;
  }

  //Division algorithm D (Knuth section 4.3.1)

  Int rem(this);
  Int d(a);
  Int dq;
  CLEAR();

  // Size
  uint32_t dSize = d.GetSize64();
  uint32_t tSize = rem.GetSize64();
  uint32_t qSize = tSize - dSize + 1;

  // D1 normalize the divisor (d!=0)
  uint32_t shift = (uint32_t)LZC(d.bits64[dSize-1]);
  d.ShiftL(shift);
  rem.ShiftL(shift);

  uint64_t  _dh    = d.bits64[dSize-1];
  uint64_t  _dl    = (dSize>1)?d.bits64[dSize-2]:0;
  int sb = tSize-1;
        
  // D2 Initialize j
  for(int j=0; j<(int)qSize; j++) {

    // D3 Estimate qhat
    uint64_t qhat = 0;
    uint64_t qrem = 0;
    int skipCorrection = false;
    uint64_t nh = rem.bits64[sb-j+1];
    uint64_t nm = rem.bits64[sb-j];

    if (nh == _dh) {
      qhat = ~0;
      qrem = nh + nm;
      skipCorrection = qrem < nh;
    } else {
      qhat = _udiv128(nh,nm,_dh,&qrem);
    }

    if (qhat == 0)
      continue;

    if (!skipCorrection) { 

      // Correct qhat
      uint64_t nl = rem.bits64[sb-j-1];

      uint64_t estProH;
      uint64_t estProL = _umul128(_dl,qhat,&estProH);
      if( isStrictGreater128(estProH,estProL,qrem,nl) ) {
        qhat--;
        qrem += _dh;
        if (qrem >= _dh) {
          estProL = _umul128(_dl,qhat,&estProH);
          if(isStrictGreater128(estProH,estProL,qrem,nl))
            qhat--;
        }
      }

    }

    // D4 Multiply and subtract    
    dq.Mult(&d,qhat);
    rem.ShiftL64BitAndSub(&dq,qSize-j-1);
    if( rem.IsNegative() ) {
      // Overflow
      rem.Add(&d);
      qhat--;
    }

    bits64[qSize-j-1] = qhat;

 }

 if( mod ) {
   // Unnormalize remainder
   rem.ShiftR(shift);
   mod->Set(&rem);
 }

}

// ------------------------------------------------

void Int::GCD(Int *a) {

    uint32_t k;
    uint32_t b;

    Int U(this);
    Int V(a);
    Int T;

    if(U.IsZero()) {
      Set(&V);
      return;
    }

    if(V.IsZero()) {
      Set(&U);
      return;
    }

    if(U.IsNegative()) U.Neg();
    if(V.IsNegative()) V.Neg();

    k = 0;
    while (U.GetBit(k)==0 && V.GetBit(k)==0)
      k++;
    U.ShiftR(k);
    V.ShiftR(k);
    if (U.GetBit(0)==1) { 
      T.Set(&V);
      T.Neg();
    } else {
      T.Set(&U);
    }

    do {

      if( T.IsNegative() ) {
        T.Neg();
        b=0;while(T.GetBit(b)==0) b++;
        T.ShiftR(b);
        V.Set(&T);
        T.Set(&U);
      } else {
        b=0;while(T.GetBit(b)==0) b++;
        T.ShiftR(b);
        U.Set(&T);
      }

      T.Sub(&V);

    } while (!T.IsZero());

    // Store gcd
    Set(&U);
    ShiftL(k); 

}

// ------------------------------------------------

void Int::SetBase10(char *value) {  

  CLEAR();
  Int pw((uint64_t)1);
  Int c;
  int lgth = (int)strlen(value);
  for(int i=lgth-1;i>=0;i--) {
    uint32_t id = (uint32_t)(value[i]-'0');
    c.Set(&pw);
    c.Mult(id);
    Add(&c);
    pw.Mult(10);
  }

}

// ------------------------------------------------

void  Int::SetBase16(char *value) {  
  SetBaseN(16,"0123456789ABCDEF",value);
}

// ------------------------------------------------

std::string Int::GetBase10() {
  return GetBaseN(10,"0123456789");
}

// ------------------------------------------------

std::string Int::GetBase16() {
  return GetBaseN(16,"0123456789ABCDEF");
}

// ------------------------------------------------

std::string Int::GetBlockStr() {
	
	char tmp[256];
	char bStr[256];
	tmp[0] = 0;
	for (int i = NB32BLOCK-3; i>=0 ; i--) {
	  sprintf(bStr, "%08X", bits[i]);
	  strcat(tmp, bStr);
	  if(i!=0) strcat(tmp, " ");
	}
	return std::string(tmp);
}

// ------------------------------------------------

std::string Int::GetC64Str(int nbDigit) {

  char tmp[256];
  char bStr[256];
  tmp[0] = '{';
  tmp[1] = 0;
  for (int i = 0; i< nbDigit; i++) {
    if (bits64[i] != 0) {
#ifdef WIN64
      sprintf(bStr, "0x%016I64XULL", bits64[i]);
#else
      sprintf(bStr, "0x%" PRIx64  "ULL", bits64[i]);
#endif
    } else {
      sprintf(bStr, "0ULL");
    }
    strcat(tmp, bStr);
    if (i != nbDigit -1) strcat(tmp, ",");
  }
  strcat(tmp,"}");
  return std::string(tmp);
}

// ------------------------------------------------

void  Int::SetBaseN(int n,char *charset,char *value) {

  CLEAR();

  Int pw((uint64_t)1);
  Int nb((uint64_t)n);
  Int c;

  int lgth = (int)strlen(value);
  for(int i=lgth-1;i>=0;i--) {
    char *p = strchr(charset,toupper(value[i]));
    if(!p) {
      printf("Invalid charset !!\n");
      return;
    }
    int id = (int)(p-charset);
    c.SetInt32(id);
    c.Mult(&pw);
    Add(&c);
    pw.Mult(&nb);

  }

}

// ------------------------------------------------

std::string Int::GetBaseN(int n,char *charset) {

  std::string ret;

  Int N(this);
  int isNegative = N.IsNegative();
  if (isNegative) N.Neg();

  // TODO: compute max digit
  unsigned char digits[1024];
  memset(digits, 0, sizeof(digits));

  int digitslen = 1;
  for (int i = 0; i < NB64BLOCK * 8; i++) {
    unsigned int carry = N.GetByte(NB64BLOCK*8 - i - 1);
    for (int j = 0; j < digitslen; j++) {
      carry += (unsigned int)(digits[j]) << 8;
      digits[j] = (unsigned char)(carry % n);
      carry /= n;
    }
    while (carry > 0) {
      digits[digitslen++] = (unsigned char)(carry % n);
      carry /= n;
    }
  }

  // reverse
  if (isNegative)
    ret.push_back('-');

  for (int i = 0; i < digitslen; i++)
    ret.push_back(charset[digits[digitslen - 1 - i]]);

  if (ret.length() == 0)
    ret.push_back('0');

  return ret;

}

// ------------------------------------------------


int Int::GetBit(uint32_t n) {

  uint32_t byte = n>>5;
  uint32_t bit  = n&31;
  uint32_t mask = 1 << bit;
  return (bits[byte] & mask)!=0;

}

// ------------------------------------------------

std::string Int::GetBase2() {

  char ret[1024];
  int k=0;

  for(int i=0;i<NB32BLOCK-1;i++) {
    unsigned int mask=0x80000000;
    for(int j=0;j<32;j++) {
      if(bits[i]&mask) ret[k]='1';
      else             ret[k]='0';
      k++;
      mask=mask>>1;
    }
  }
  ret[k]=0;

  return std::string(ret);

}

bool Int::IsProbablePrime() {

  // Prime cheking (probalistic Miller-Rabin test)
  Int::SetupField(this);
  int nbBit = GetBitLength();

  Int Q(this);
  Q.SubOne();
  Int N1(&Q);
  uint64_t e = 0;
  while(Q.IsEven()) {
    Q.ShiftR(1);
    e++;
  }

  uint64_t k = 50;

  for(uint64_t i = 0; i < k; i++) {

    Int a;
    Int x;
    x.SetInt32(0); 
    while(x.IsLowerOrEqual(&_ONE) || x.IsGreaterOrEqual(&N1))
      x.Rand(nbBit);
    x.ModExp(&Q);
    if(x.IsOne() || x.IsEqual(&N1))
      continue;

    for(uint64_t j = 0; j < e - 1; j++) {
      x.ModSquare(&x);
      if(x.IsOne()) {
        // Composite
        return false;
      }
      if(x.IsEqual(&N1))
        break;
    }

    if(x.IsEqual(&N1))
      continue;

    return false;

  }

  // Probable prime
  return true;

}

extern int64_t iCountMax;
extern int64_t iCountTotal;
extern int64_t iCountHist[12];

// ------------------------------------------------

bool Int::CheckInv(Int* a) {

  Int b(a);
  Int c;
  bool ok = true;

  b.ModInv();
  c = b;
  b.ModMul(a);
  if(!b.IsOne()) {

    Int e(Int::GetFieldCharacteristic());
    e.Sub(2ULL);
    Int g(a);
    g.ModExp(&e);

    printf("ModInv() Results Wrong for %s\n",a->GetBase16().c_str());
    printf(" Got: %s\n",c.GetBase16().c_str());
    printf(" Exp: %s\n",g.GetBase16().c_str());
    ok = false;

  }

  if(ok) {
    b = c;
    c.ModInv();
    if(!c.IsEqual(a)) {

      Int e(Int::GetFieldCharacteristic());
      e.Sub(2ULL);
      Int g(&b);
      g.ModExp(&e);

      printf("ModInv() Results Wrong for %s\n",b.GetBase16().c_str());
      printf(" Got: %s\n",c.GetBase16().c_str());
      printf(" Exp: %s\n",g.GetBase16().c_str());
      ok = false;
    }
  }

  return ok;

}

extern uint64_t totalCount;

void Int::Check() {

  double t0;
  double t1;
  double tTotal;
  int   i;
  bool ok;

  Int a, b, c, d, e, f, R;

  a.SetBase10("4743256844168384767987");
  b.SetBase10("1679314142928575978367");
  if (strcmp(a.GetBase10().c_str(), "4743256844168384767987") != 0) {
    printf(" GetBase10() failed ! %s!=4743256844168384767987\n", a.GetBase10().c_str());
  }
  if (strcmp(b.GetBase10().c_str(), "1679314142928575978367") != 0) {
    printf(" GetBase10() failed ! %s!=1679314142928575978367\n", b.GetBase10().c_str());
    return;
  }

  printf("GetBase10() Results OK\n");

  // Add -------------------------------------------------------------------------------------------
  t0 = Timer::get_tick();
  for (i = 0; i < 10000; i++) c.Add(&a, &b);
  t1 = Timer::get_tick();

  if (c.GetBase10() == "6422570987096960746354") {
    printf("Add() Results OK : ");
    Timer::printResult("Add", 10000, t0, t1);
  } else {
    printf("Add() Results Wrong\nR=%s\nT=6422570987096960746354\n", c.GetBase10().c_str());
    return;
  }

  // Mult -------------------------------------------------------------------------------------------
  a.SetBase10("3890902718436931151119442452387018319292503094706912504064239834754167");
  b.SetBase10("474325684416838476798716793141429285759783676422570987096960746354");
  e.SetBase10("1845555094921934741640873731771879197054909502699192730283220486240724687661257894226660948002650341240452881231721004292250660431557118");

  t0 = Timer::get_tick();
  for (i = 0; i < 10000; i++) c.Mult(&a, &b);
  t1 = Timer::get_tick();

  if (c.IsEqual(&e)) {
    printf("Mult() Results OK : ");
    Timer::printResult("Mult", 10000, t0, t1);
  } else {
    printf("Mult() Results Wrong\nR=%s\nT=%s\n",e.GetBase10().c_str(), c.GetBase10().c_str());
    return;
  }
  
  // Div -------------------------------------------------------------------------------------------
  tTotal = 0.0;
  ok = true;
  for (int i = 0; i < 1000 && ok; i++) {

    a.Rand(BISIZE);
    b.Rand(BISIZE/2);
    d.Set(&a);
    e.Set(&b);

    t0 = Timer::get_tick();
    a.Div(&b, &c);
    t1 = Timer::get_tick();
    tTotal += (t1 - t0);

    f.Set(&a);
    a.Mult(&e);
    a.Add(&c);
    if (!a.IsEqual(&d)) {
	    ok = false;
      printf("Div() Results Wrong \nN: %s\nD: %s\nQ: %s\nR: %s\nCheck: %s\n", 
        d.GetBase16().c_str(),
        b.GetBase16().c_str(),
        f.GetBase16().c_str(),
        c.GetBase16().c_str(),
        a.GetBase16().c_str()
      );
      return;
    }

  }
  
  if(ok) {
    printf("Div() Results OK : ");
    Timer::printResult("Div", 1000, 0, tTotal);
  }

  // Modular arithmetic -------------------------------------------------------------------------------
  // SecpK1 prime
  b.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
  //b.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
  //b.SetBase16("7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFED");
  //b.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDC7");
  Int::SetupField(&b);
  printf("R1=%s\n",Int::GetR()->GetBase16().c_str());
  printf("R2=%s\n",Int::GetR2()->GetBase16().c_str());

  // Check work only for prime close to a power of 2
  int pSize = Int::GetFieldCharacteristic()->GetBitLength();
  printf("Field characteristic size: %dbits\n",pSize);

  // ModInv -------------------------------------------------------------------------------------------

  ok = true;
  for (int i = 0; i < 10000 && ok; i++) {

    // Euler a^-1 = a^(p-2) mod p (p is prime)
    Int e(Int::GetFieldCharacteristic());
    e.Sub(2ULL);
    a.Rand(pSize);
    b = a;
    b.ModExp(&e);

    a.ModInv();
    if (!a.IsEqual(&b)) {
      ok =false;
    }

  }

  if (!ok) {
    printf("ModInv()/ModExp() Results Wrong:\nModInv=%s\nModExp=%s\n", a.GetBase16().c_str(),b.GetBase16().c_str());
    return;
  } else {
    printf("ModInv()/ModExp() Results OK\n");
  }

  a.SetInt32(0);
  a.ModInv();
  if(!a.IsZero()) {
    printf("ModInv(0) does not return 0!\n");
  }

  a.Set(&_ONE);
  for(int64_t i = 0; i < pSize - 1 && ok; i++) {
    ok = CheckInv(&a);
    b = a;
    b.ModNeg();
    ok = CheckInv(&b);
    a.ShiftL(1);
  }

  a.Set(&_ONE);
  for(int64_t i = 0; i < pSize - 1 && ok; i++) {
    ok = CheckInv(&a);
    b = a;
    b.ModNeg();
    ok = CheckInv(&b);
    a.ShiftL(1);
    if(i%2==0 && i>0)
      a.AddOne();
  }

  a.Set(Int::GetFieldCharacteristic());
  for(int64_t i = 0; i < 100000 && ok; i++) {
    a.SubOne();
    ok = CheckInv(&a);
  }
  a.Set(&_ONE);
  for(int64_t i = 0; i < 100000 && ok; i++) {
    ok = CheckInv(&a);
    a.AddOne();
  }

  if(ok)
    printf("ModInv() Edge cases Results OK\n");
  else
    printf("ModInv() Edge cases Results Wrong\n");

  totalCount = 0;

  for(int64_t i = 0; i <= 100000 && ok; i++) {
    a.Rand(pSize);
    ok = CheckInv(&a);
    if(i%1000000==0) printf(".");
  }

  printf("Avg = %.2f\n",(double)totalCount/200000.0);

  a.Rand(pSize);
  b.Rand(pSize-64);
  t0 = Timer::get_tick();
  uint64_t c0 = __rdtsc();
  for (int i = 0; i < 400000; i++) {
    a.Add(&b);
    a.ModInv();
  }
  uint64_t c1 = __rdtsc();
  t1 = Timer::get_tick();

  printf("ModInv() Results OK : ");
  Timer::printResult("Inv", 400000, 0, t1 - t0);
  printf("ModInv() cycles : %.2f\n",(double)(c1-c0)/400000.0);
  double movInvCost = (t1-t0);

  // ModSqrt ------------------------------------------------------------------------------------

  ok = true;
  for(int i = 0; i < 1000 && ok; i++) {

    bool hasSqrt = false;
    while(!hasSqrt) {
      a.Rand(pSize);
      hasSqrt = !a.IsZero() && a.IsLower(Int::GetFieldCharacteristic()) && a.HasSqrt();
    }

    c.Set(&a);
    a.ModSqrt();
    b.ModSquare(&a);
    if(!b.IsEqual(&c)) {
      printf("ModSqrt() wrong !\n");
      ok = false;
    }

  }
  if(!ok) return;

  printf("ModSqrt() Results OK !\n");

  // Check of the Secp256K1 specific part
  b.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
  if( Int::GetFieldCharacteristic()->IsEqual(&b) ) {

    // IntGroup -----------------------------------------------------------------------------------
    Int m[256];
    Int chk[256];
    IntGroup g(256);

    g.Set(m);
    for(int i = 0; i < 256; i++) {
      m[i].Rand(pSize);
      chk[i].Set(m + i);
      chk[i].ModInv();
    }
    g.ModInv();
    ok = true;
    for(int i = 0; i < 256; i++) {
      if(!m[i].IsEqual(chk + i)) {
        ok = false;
        printf("IntGroup.ModInv() Wrong !\n");
        printf("[%d] %s\n",i,m[i].GetBase16().c_str());
        printf("[%d] %s\n",i,chk[i].GetBase16().c_str());
        return;
      }
    }

    t0 = Timer::get_tick();
    for(int j = 0; j < 1000; j++) {
      for(int i = 0; i < 256; i++) {
        m[i].Rand(pSize);
      }
      g.ModInv();
    }
    t1 = Timer::get_tick();

    printf("IntGroup.ModInv() Results OK : ");
    Timer::printResult("Inv",1000 * 256,0,t1 - t0);

    // ModMulK1 ------------------------------------------------------------------------------------

    for(int i = 0; i < 100000; i++) {
      a.Rand(pSize);
      b.Rand(pSize);
      c.ModMul(&a,&b);
      d.ModMulK1(&a,&b);
      if(!c.IsEqual(&d)) {
        printf("ModMulK1() Wrong !\n");
        printf("[%d] %s\n",i,c.GetBase16().c_str());
        printf("[%d] %s\n",i,d.GetBase16().c_str());
        return;
      }
    }

    a.Rand(pSize);
    b.Rand(pSize);
    t0 = Timer::get_tick();
    for(int i = 0; i < 1000000; i++) {
      a.AddOne();
      b.AddOne();
      c.ModMulK1(&a,&b);
    }
    t1 = Timer::get_tick();

    printf("ModMulK1() Results OK : ");
    Timer::printResult("Mult",1000000,0,t1 - t0);

    // ModSqrK1 ------------------------------------------------------------------------------------

    for(int i = 0; i < 100000; i++) {
      a.Rand(pSize);
      c.ModMul(&a,&a);
      d.ModSquareK1(&a);
      if(!c.IsEqual(&d)) {
        printf("ModSquareK1() Wrong !\n");
        printf("[%d] %s\n",i,c.GetBase16().c_str());
        printf("[%d] %s\n",i,d.GetBase16().c_str());
        return;
      }
    }

    a.Rand(pSize);
    b.Rand(pSize);
    t0 = Timer::get_tick();
    for(int i = 0; i < 1000000; i++) {
      a.AddOne();
      b.AddOne();
      c.ModSquareK1(&b);
    }
    t1 = Timer::get_tick();

    printf("ModSquareK1() Results OK : ");
    Timer::printResult("Sqr",1000000,0,t1 - t0);

    // modInvCost is for 200000 iterations
    double cost = movInvCost * 5.0 / (t1 - t0);
    printf("ModInv() Cost : %.1f S\n",cost);

    // ModMulK1 order -----------------------------------------------------------------------------
    // InitK1() is done by secpK1
    b.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
    Int::SetupField(&b);

    for(int i = 0; i < 100000; i++) {
      a.Rand(pSize);
      b.Rand(pSize);
      c.ModMul(&a,&b);
      d.Set(&a);
      d.ModMulK1order(&b);
      if(!c.IsEqual(&d)) {
        printf("ModMulK1order() Wrong !\n");
        printf("[%d] %s\n",i,c.GetBase16().c_str());
        printf("[%d] %s\n",i,d.GetBase16().c_str());
        return;
      }
    }

    t0 = Timer::get_tick();
    for(int i = 0; i < 1000000; i++) {
      a.Rand(pSize);
      b.Rand(pSize);
      c.Set(&a);
      c.ModMulK1order(&b);
    }
    t1 = Timer::get_tick();

    printf("ModMulK1order() Results OK : ");
    Timer::printResult("Mult",1000000,0,t1 - t0);

  }

  // Restore Secp256K1 prime
  b.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
  Int::SetupField(&b);

}
