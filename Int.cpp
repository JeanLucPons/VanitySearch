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

#include "Int.h"
#include "IntGroup.h"
#include <string.h>
#include <emmintrin.h>
#include "Timer.h"

#define MAX(x,y) (((x)>(y))?(x):(y))
#define MIN(x,y) (((x)<(y))?(x):(y))

Int _ONE(1);


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

	volatile unsigned char c=0;
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

void Int::ShiftL32BitAndSub(Int *a,int n) {

  Int b;
  int i=NB32BLOCK-1;

  for(;i>=n;i--)
    b.bits[i] = ~a->bits[i-n];
  for(;i>=0;i--)
    b.bits[i] = 0xFFFFFFFF;

  Add(&b);
  AddOne();

}

// ------------------------------------------------

void Int::ShiftL(uint32_t n) {
    
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

void Int::Mult(Int *a) {

  Int b(this);
  Mult(a,&b);

}

// ------------------------------------------------

void Int::IMult(int64_t a) {

	// Make a positive
	if (a < 0LL) {
		a = -a;
		Neg();
	}

	imm_mul(bits64, a, bits64);

}

// ------------------------------------------------

void Int::Mult(uint64_t a) {

	imm_mul(bits64, a, bits64);

}
// ------------------------------------------------

void Int::IMult(Int *a, int64_t b) {
  
  Set(a);

  // Make b positive
  if (b < 0LL) {
	Neg();
	b = -b;
  }
  imm_mul(bits64, b, bits64);

}

// ------------------------------------------------

void Int::Mult(Int *a, uint64_t b) {

  imm_mul(a->bits64, b, bits64);

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

void Int::Mult(Int *a,uint32_t b) {
  imm_mul(a->bits64, (uint64_t)b, bits64);
}

// ------------------------------------------------

static uint32_t bitLength(uint32_t dw) {
  
  uint32_t mask = 0x80000000;
  uint32_t b=0;
  while(b<32 && (mask & dw)==0) {
    b++;
    mask >>= 1;
  }
  return b;

}

// ------------------------------------------------

int Int::GetBitLength() {

  Int t(this);
  if(IsNegative())
	  t.Neg();

  int i=NB32BLOCK-1;
  while(i>=0 && t.bits[i]==0) i--;
  if(i<0) return 0;
  return (32-bitLength(t.bits[i])) + i*32;

}

// ------------------------------------------------

int Int::GetSize() {

  int i=NB32BLOCK-1;
  while(i>0 && bits[i]==0) i--;
  return i+1;

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
  uint32_t dSize = d.GetSize();
  uint32_t tSize = rem.GetSize();
  uint32_t qSize = tSize - dSize + 1;

  // D1 normalize the divisor
  uint32_t shift = bitLength(d.bits[dSize-1]);
  if (shift > 0) {
    d.ShiftL(shift);
    rem.ShiftL(shift);
  }

  uint32_t  _dh    = d.bits[dSize-1];
  uint64_t  dhLong = _dh;
  uint32_t  _dl    = (dSize>1)?d.bits[dSize-2]:0;
  int sb = tSize-1;
        
  // D2 Initialize j
  for(int j=0; j<(int)qSize; j++) {

    // D3 Estimate qhat
    uint32_t qhat = 0;
    uint32_t qrem = 0;
    int skipCorrection = false;
    uint32_t nh = rem.bits[sb-j+1];
    uint32_t nm = rem.bits[sb-j];

    if (nh == _dh) {
      qhat = ~0;
      qrem = nh + nm;
      skipCorrection = qrem < nh;
    } else {
      uint64_t nChunk = ((uint64_t)nh << 32) | (uint64_t)nm;
      qhat = (uint32_t) (nChunk / dhLong);
      qrem = (uint32_t) (nChunk % dhLong);
    }

    if (qhat == 0)
      continue;

    if (!skipCorrection) { 

      // Correct qhat
      uint64_t nl = (uint64_t)rem.bits[sb-j-1];
      uint64_t rs = ((uint64_t)qrem << 32) | nl;
      uint64_t estProduct = (uint64_t)_dl * (uint64_t)(qhat);

      if (estProduct>rs) {
        qhat--;
        qrem = (uint32_t)(qrem + (uint32_t)dhLong);
        if ((uint64_t)qrem >= dhLong) {
          estProduct = (uint64_t)_dl * (uint64_t)(qhat);
          rs = ((uint64_t)qrem << 32) | nl;
          if(estProduct>rs)
            qhat--;
        }
      }

    }

    // D4 Multiply and subtract    
    dq.Mult(&d,qhat);
    rem.ShiftL32BitAndSub(&dq,qSize-j-1);
    if( rem.IsNegative() ) {
      // Overflow
      rem.Add(&d);
      qhat--;
    }

    bits[qSize-j-1] = qhat;

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
  Int pw(1);
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

  Int pw((uint32_t)1);
  Int nb((int32_t)n);
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


// ------------------------------------------------

void Int::Check() {

  double t0;
  double t1;
  double tTotal;
  int   i;
  bool ok;

  Int a, b, c, d, e, R;

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

    a.Mult(&e);
    a.Add(&c);
    if (!a.IsEqual(&d)) {
	  ok = false;
      printf("Div() Results Wrong \nN: %s\nD: %s\nQ: %s\nR: %s\n", 
        d.GetBase16().c_str(),
        b.GetBase16().c_str(),
        a.GetBase16().c_str(),
        c.GetBase16().c_str()
        
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
  Int::SetupField(&b);

  // ModInv -------------------------------------------------------------------------------------------

  for (int i = 0; i < 1000 && ok; i++) {
    a.Rand(BISIZE);
    b = a;
    a.ModInv();
    a.ModMul(&b);
    if (!a.IsOne()) {
      printf("ModInv() Results Wrong [%d] %s\n",i, a.GetBase16().c_str());
	  ok = false;
    }
  }

  ok = true;
  for (int i = 0; i < 100 && ok; i++) {

    // Euler a^-1 = a^(p-2) mod p (p is prime)
    Int e(Int::GetFieldCharacteristic());
    e.Sub(2ULL);
    a.Rand(BISIZE);
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

  t0 = Timer::get_tick();
  a.Rand(BISIZE);
  for (int i = 0; i < 100000; i++) {
    a.AddOne();
    a.ModInv();
  }
  t1 = Timer::get_tick();

  printf("ModInv() Results OK : ");
  Timer::printResult("Inv", 100000, 0, t1 - t0);

  // IntGroup -----------------------------------------------------------------------------------

  Int m[256];
  Int chk[256];
  IntGroup g(256);

  g.Set(m);
  for (int i = 0; i < 256; i++) {
    m[i].Rand(256);
    chk[i].Set(m + i);
    chk[i].ModInv();
  }
  g.ModInv();
  ok = true;
  for (int i = 0; i < 256; i++) {
    if (!m[i].IsEqual(chk + i)) {
      ok = false;
      printf("IntGroup.ModInv() Wrong !\n");
      printf("[%d] %s\n", i, m[i].GetBase16().c_str());
      printf("[%d] %s\n", i, chk[i].GetBase16().c_str());
      return;
    }
  }

  for (int i = 0; i < 256; i++)
    m[i].Rand(256);
  t0 = Timer::get_tick();
  for (int j = 0; j < 1000; j++) {
    for (int i = 0; i < 256; i++)
      m[i].AddOne();
    g.ModInv();
  }
  t1 = Timer::get_tick();

  printf("IntGroup.ModInv() Results OK : ");
  Timer::printResult("Inv", 1000 * 256, 0, t1 - t0);

  // ModMulK1 ------------------------------------------------------------------------------------

  for (int i = 0; i < 100000; i++) {
    a.Rand(BISIZE);
    b.Rand(BISIZE);
    c.ModMul(&a,&b);
    d.ModMulK1(&a,&b);
    if (!c.IsEqual(&d)) {
      printf("ModMulK1() Wrong !\n");
      printf("[%d] %s\n", i, c.GetBase16().c_str());
      printf("[%d] %s\n", i, d.GetBase16().c_str());
      return;
    }
  }

  t0 = Timer::get_tick();
  a.Rand(BISIZE);
  for (int i = 0; i < 1000000; i++) {
    a.AddOne();
    c.ModMulK1(&a);
  }
  t1 = Timer::get_tick();

  printf("ModMulK1() Results OK : ");
  Timer::printResult("Mult", 1000000, 0, t1 - t0);

  // ModSqrK1 ------------------------------------------------------------------------------------

  for (int i = 0; i < 100000; i++) {
    a.Rand(BISIZE);
    c.ModMul(&a, &a);
    d.ModSquareK1(&a);
    if (!c.IsEqual(&d)) {
      printf("ModSquareK1() Wrong !\n");
      printf("[%d] %s\n", i, c.GetBase16().c_str());
      printf("[%d] %s\n", i, d.GetBase16().c_str());
      return;
    }
  }

  t0 = Timer::get_tick();
  b.Rand(BISIZE);
  for (int i = 0; i < 1000000; i++) {
    b.AddOne();
    c.ModSquareK1(&b);
  }
  t1 = Timer::get_tick();

  printf("ModSquareK1() Results OK : ");
  Timer::printResult("Mult", 1000000, 0, t1 - t0);

  // ModMulK1 order -----------------------------------------------------------------------------
  // InitK1() is done by secpK1
  b.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
  Int::SetupField(&b);

  for (int i = 0; i < 100000; i++) {
    a.Rand(BISIZE);
    b.Rand(BISIZE);
    c.ModMul(&a,&b);
    d.Set(&a);
    d.ModMulK1order(&b);
    if (!c.IsEqual(&d)) {
      printf("ModMulK1order() Wrong !\n");
      printf("[%d] %s\n", i, c.GetBase16().c_str());
      printf("[%d] %s\n", i, d.GetBase16().c_str());
      return;
    }
  }

  a.Rand(BISIZE);
  b.Rand(BISIZE);
  t0 = Timer::get_tick();
  for (int i = 0; i < 1000000; i++) {
    c.Set(&a);
    b.AddOne();
    c.ModMulK1order(&b);
  }
  t1 = Timer::get_tick();

  printf("ModMulK1order() Results OK : ");
  Timer::printResult("Mult", 1000000, 0, t1 - t0);

  // ModSqrt ------------------------------------------------------------------------------------
  b.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
  Int::SetupField(&b);

  ok = true;
  for (int i = 0; i < 100 && ok; i++) {

    bool hasSqrt = false;
    while (!hasSqrt) {
      a.Rand(BISIZE);
      hasSqrt = !a.IsZero() && a.IsLower(Int::GetFieldCharacteristic()) && a.HasSqrt();
    }

    c.Set(&a);
    a.ModSqrt();
    b.ModSquare(&a);
    if (!b.IsEqual(&c)) {
      printf("ModSqrt() wrong !\n");
      ok = false;
    }

  }
  if(!ok) return;

  printf("ModSqrt() Results OK !\n");

}
