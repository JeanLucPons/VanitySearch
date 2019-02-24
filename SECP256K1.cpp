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

#include "SECP256k1.h"
#include "hash/sha256.h"
#include "hash/ripemd160.h"
#include "Base58.h"
#include <string.h>

#ifdef WIN64
#include <windows.h>
#else
typedef int HANDLE;
#endif

Secp256K1::Secp256K1() {
}

void Secp256K1::Init() {

  // Prime for the finite field
  Int P;
  P.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");

  // Set up field
  Int::SetupField(&P);

  // Generator point and order
  G.x.SetBase16("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
  G.y.SetBase16("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
  G.z.SetInt32(1);
  order.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

  // Compute Generator table
  Point N(G);
  for(int i = 0; i < 256; i++) {
    GTable[i] = N;
    N = DoubleDirect(N);
  }

}

Secp256K1::~Secp256K1() {
}

void PrintResult(HANDLE H,bool ok) {
#ifdef WIN64
  if(ok) {
    SetConsoleTextAttribute(H,10);
    printf("OK\n");
    SetConsoleTextAttribute(H,7);
  }
  else {
    SetConsoleTextAttribute(H,12);
    printf("Failed !\n");
    SetConsoleTextAttribute(H,7);
  }
#else
  if(ok) {
    printf("OK\n");
  }
  else {
    printf("Failed !\n");
  }
#endif
}

void CheckAddress(Secp256K1 *T,HANDLE H,std::string address,std::string privKeyStr) {

  Int privKey = T->DecodePrivateKey((char *)privKeyStr.c_str());
  Point pub = T->ComputePublicKey(&privKey);
  std::string calcAddress = T->GetAddress(pub,false);
  std::string calcAddressComp = T->GetAddress(pub,true);

  printf("Adress : %s ",address.c_str());

#ifdef WIN64
  if(address == calcAddress) {
    SetConsoleTextAttribute(H,10);
    printf("OK!\n");
    SetConsoleTextAttribute(H,7);
    return;
  }

  if(address == calcAddressComp) {
    SetConsoleTextAttribute(H,10);
    printf("OK(comp)!\n");
    SetConsoleTextAttribute(H,7);
    return;
  }

  SetConsoleTextAttribute(H,12);
  printf("Failed ! %s\n",calcAddress.c_str());
  SetConsoleTextAttribute(H,7);
#else
  if(address == calcAddress) {
    printf("OK!\n");
    return;
  }

  if(address == calcAddressComp) {
    printf("OK(comp)!\n");
    return;
  }

  printf("Failed ! %s\n",calcAddress.c_str());
#endif

}

void Secp256K1::Check() {

#ifdef WIN64
  HANDLE H = GetStdHandle(STD_OUTPUT_HANDLE);
#else
  HANDLE H = 0;
#endif

  printf("Check Generator :");

  bool ok = true;
  int i = 0;
  while(i < 256 && EC(GTable[i])) {
    i++;
  }
  PrintResult(H,i == 256);

  printf("Check Double :");
  Point Pt(G);
  Point R1;
  Point R2;
  Point R3;
  R1 = Double(G);
  R1.Reduce();
  PrintResult(H,EC(R1));

  printf("Check Add :");
  R2 = Add(G,R1);
  R3 = Add(R1,R2);
  R3.Reduce();
  PrintResult(H,EC(R3));

  printf("Check GenKey :");
  Int privKey;
  privKey.SetBase16("46b9e861b63d3509c88b7817275a30d22d62c8cd8fa6486ddee35ef0d8e0495f");
  Point pub = ComputePublicKey(&privKey);
  Point expectedPubKey;
  expectedPubKey.x.SetBase16("2500e7f3fbddf2842903f544ddc87494ce95029ace4e257d54ba77f2bc1f3a88");
  expectedPubKey.y.SetBase16("37a9461c4f1c57fecc499753381e772a128a5820a924a2fa05162eb662987a9f");
  expectedPubKey.z.SetInt32(1);

  PrintResult(H,pub.equals(expectedPubKey));

  CheckAddress(this,H,"15t3Nt1zyMETkHbjJTTshxLnqPzQvAtdCe","5HqoeNmaz17FwZRqn7kCBP1FyJKSe4tt42XZB7426EJ2MVWDeqk");
  CheckAddress(this,H,"1BoatSLRHtKNngkdXEeobR76b53LETtpyT","5J4XJRyLVgzbXEgh8VNi4qovLzxRftzMd8a18KkdXv4EqAwX3tS");
  CheckAddress(this,H,"1JeanLucgidKHxfY5gkqGmoVjo1yaU4EDt","5JHMHsrrLNoEnPacZJ6VCv2YraHxPhApEp2DEa1uFzGndiByVzV");
  CheckAddress(this,H,"1Test6BNjSJC5qwYXsjwKVLvz7DpfLehy","5HytzR8p5hp8Cfd8jsVFnwMNXMsEW1sssFxMQYqEUjGZN72iLJ2");
  CheckAddress(this,H,"1BitcoinP7vnLpsUHWbzDALyJKnNo16Qms","5HwKrREWhgtupmZH9cE1wFvHQJhbXMxm28L5KaVhtReBKGXL2J1");

  // 1ViViGLEawN27xRzGrEhhYPQrZiTKvKLo
  pub.x.SetBase16(/*04*/"75249c39f38baa6bf20ab472191292349426dc3652382cdc45f65695946653dc");
  pub.y.SetBase16("978b2659122fe1df1be132167f27b74e5d4a2f3ecbbbd0b3fbcc2f4983518674");
  printf("Check Calc PubKey (full) %s :",GetAddress(pub,false).c_str());
  PrintResult(H,EC(pub));

  // 1Gp7rQ4GdooysEAEJAS2o4Ktjvf1tZCihp
  pub.x.SetBase16(/*02*/"2b70d6a249aeb187d6f079ecc0fb34d075056ca985384240166a2080c7d2beb5");
  pub.y = GetY(pub.x,true);
  printf("Check Calc PubKey (even) %s:",GetAddress(pub,true).c_str());
  PrintResult(H,EC(pub));

  // 18aPiLmTow7Xgu96msrDYvSSWweCvB9oBA
  pub.x.SetBase16(/*03*/"3bf3d80f868fa33c6353012cb427e98b080452f19b5c1149ea2acfe4b7599739");
  pub.y = GetY(pub.x,false);
  printf("Check Calc PubKey (odd) %s:",GetAddress(pub,true).c_str());
  PrintResult(H,EC(pub));

}


Point Secp256K1::ComputePublicKey(Int *privKey) {

  Point Q;
  Q.Clear();
  for(int i = 0; i < 256; i++) {
    if(privKey->GetBit(i))
      if(Q.isZero())
        Q = GTable[i];
      else
        Q = Add(Q,GTable[i]);
  }
  Q.Reduce();
  return Q;

}

Point Secp256K1::NextKey(Point &key) {
  // Input key must be reduced and different from G
  // in order to use AddDirect
  return AddDirect(key,G);
}

Int Secp256K1::DecodePrivateKey(char *key) {

  Int ret;
  std::vector<unsigned char> privKey;

  if(key[0] == '5') {

    // Not compressed
    DecodeBase58(key,privKey);
    if(privKey.size() != 37) {
      printf("Invalid private key, size != 37 (size=%d)!\n",(int)privKey.size());
      ret.SetInt32(-1);
      return ret;
    }

    if(privKey[0] != 0x80) {
      printf("Invalid private key, wrong prefix !\n");
      return ret;
    }

    int count = 31;
    for(int i = 1; i < 33; i++)
      ret.SetByte(count--,privKey[i]);

    return ret;

  } else if(key[0] == 'K' || key[0] == 'L') {

    // Compressed
    DecodeBase58(key,privKey);
    if(privKey.size() != 38) {
      printf("Invalid private key, size != 38 (size=%d)!\n",(int)privKey.size());
      ret.SetInt32(-1);
      return ret;
    }

    int count = 31;
    for(int i = 1; i < 33; i++)
      ret.SetByte(count--,privKey[i]);

    return ret;

  }

  printf("Invalid private key, not starting with 5,K or L !\n");
  return ret;

}

void Secp256K1::GetHash160(bool compressed,
  Point &k0,Point &k1,Point &k2,Point &k3,
  uint8_t *h0,uint8_t *h1,uint8_t *h2,uint8_t *h3) {

  unsigned char sh0[64];
  unsigned char sh1[64];
  unsigned char sh2[64];
  unsigned char sh3[64];

  if(!compressed) {

    unsigned char b0[128];
    unsigned char b1[128];
    unsigned char b2[128];
    unsigned char b3[128];

    b0[0] = 0x4;
    b1[0] = 0x4;
    b2[0] = 0x4;
    b3[0] = 0x4;

    k0.x.Get32Bytes(b0 + 1);
    k0.y.Get32Bytes(b0 + 33);
    k1.x.Get32Bytes(b1 + 1);
    k1.y.Get32Bytes(b1 + 33);
    k2.x.Get32Bytes(b2 + 1);
    k2.y.Get32Bytes(b2 + 33);
    k3.x.Get32Bytes(b3 + 1);
    k3.y.Get32Bytes(b3 + 33);

    sha256sse_65(b0,b1,b2,b3,sh0,sh1,sh2,sh3);
    ripemd160sse_32(sh0,sh1,sh2,sh3,h0,h1,h2,h3);

  } else {

    unsigned char b0[64];
    unsigned char b1[64];
    unsigned char b2[64];
    unsigned char b3[64];

    b0[0] = k0.y.IsEven() ? 0x2 : 0x3;
    b1[0] = k1.y.IsEven() ? 0x2 : 0x3;
    b2[0] = k2.y.IsEven() ? 0x2 : 0x3;
    b3[0] = k3.y.IsEven() ? 0x2 : 0x3;

    k0.x.Get32Bytes(b0 + 1);
    k1.x.Get32Bytes(b1 + 1);
    k2.x.Get32Bytes(b2 + 1);
    k3.x.Get32Bytes(b3 + 1);

    sha256sse_33(b0,b1,b2,b3,sh0,sh1,sh2,sh3);
    ripemd160sse_32(sh0,sh1,sh2,sh3,h0,h1,h2,h3);

  }



}

void Secp256K1::GetHash160(Point &pubKey,bool compressed,unsigned char *hash) {

  unsigned char publicKeyBytes[128];
  unsigned char shapk[64];

  if(!compressed) {

    // Full public key
    publicKeyBytes[0] = 0x4;
    pubKey.x.Get32Bytes(publicKeyBytes + 1);
    pubKey.y.Get32Bytes(publicKeyBytes + 33);
    sha256(publicKeyBytes,65,shapk);

  } else {

    // Compressed public key
    publicKeyBytes[0] = pubKey.y.IsEven() ? 0x2 : 0x3;
    pubKey.x.Get32Bytes(publicKeyBytes + 1);
    sha256_33(publicKeyBytes,shapk);

  }

  ripemd160_32(shapk,hash);

}

std::string Secp256K1::GetPrivAddress(Int &privKey) {

  unsigned char adress[37];

  adress[0] = 0x80;
  privKey.Get32Bytes(adress + 1);

  // Compute checksum
  unsigned char sha1[32];
  unsigned char sha2[32];
  sha256(adress,33,sha1);
  sha256(sha1,32,sha2);
  adress[33] = sha2[0];
  adress[34] = sha2[1];
  adress[35] = sha2[2];
  adress[36] = sha2[3];

  return EncodeBase58(adress,adress + 37);

}

std::string Secp256K1::GetAddress(unsigned char *hash20,bool compressed) {

  unsigned char adress[25];
  adress[0] = 0; // Version
  memcpy(adress + 1,hash20,20);

  // Compute checksum
  unsigned char sha1[32];
  unsigned char sha2[32];
  sha256(adress,21,sha1);
  sha256(sha1,32,sha2);
  adress[21] = sha2[0];
  adress[22] = sha2[1];
  adress[23] = sha2[2];
  adress[24] = sha2[3];

  // Base58
  return EncodeBase58(adress,adress + 25);

}

std::string Secp256K1::GetAddress(Point &pubKey,bool compressed) {

  unsigned char adress[25];

  adress[0] = 0; // Version
  GetHash160(pubKey,compressed,adress + 1);

  // Compute checksum
  unsigned char sha1[32];
  unsigned char sha2[32];
  sha256(adress,21,sha1);
  sha256(sha1,32,sha2);
  adress[21] = sha2[0];
  adress[22] = sha2[1];
  adress[23] = sha2[2];
  adress[24] = sha2[3];

  // Base58
  return EncodeBase58(adress,adress + 25);

}

bool Secp256K1::CheckPudAddress(std::string address) {

  std::vector<unsigned char> pubKey;
  DecodeBase58(address,pubKey);

  if(pubKey.size()!=25)
    return false;

  // Check checksum
  unsigned char sha1[32];
  unsigned char sha2[32];
  sha256(pubKey.data(),21,sha1);
  sha256(sha1,32,sha2);
  
  return  (pubKey[21] == sha2[0]) &&
          (pubKey[22] == sha2[1]) &&
          (pubKey[23] == sha2[2]) &&
          (pubKey[24] == sha2[3]);

}

Point Secp256K1::AddDirect(Point &p1,Point &p2) {

  Int _s;
  Int _p;
  Int dy;
  Int dx;
  Point r;
  r.z.SetInt32(1);

  dy.ModSub(&p2.y,&p1.y);
  dx.ModSub(&p2.x,&p1.x);
  dx.ModInv();
  _s.ModMul(&dy,&dx);     // s = (p2.y-p1.y)*inverse(p2.x-p1.x);

  _p.ModSquare(&_s);       // _p = pow2(s)

  r.x.ModSub(&_p,&p1.x);
  r.x.ModSub(&p2.x);       // rx = pow2(s) - p1.x - p2.x;

  r.y.ModSub(&p2.x,&r.x);
  r.y.ModMul(&_s);
  r.y.ModSub(&p2.y);       // ry = - p2.y - s*(ret.x-p2.x);  

  return r;

}

Point Secp256K1::Add(Point &p1,Point &p2) {

  Int u;
  Int v;
  Int u1;
  Int u2;
  Int v1;
  Int v2;
  Int vs2;
  Int vs3;
  Int us2;
  Int w;
  Int a;
  Int us2w;
  Int vs2v2;
  Int vs3u2;
  Int _2vs2v2;
  Int x3;
  Int vs3y1;
  Point r;

  /*
  U1 = Y2 * Z1
  U2 = Y1 * Z2
  V1 = X2 * Z1
  V2 = X1 * Z2
  if (V1 == V2)
    if (U1 != U2)
      return POINT_AT_INFINITY
    else
      return POINT_DOUBLE(X1, Y1, Z1)
  U = U1 - U2
  V = V1 - V2
  W = Z1 * Z2
  A = U ^ 2 * W - V ^ 3 - 2 * V ^ 2 * V2
  X3 = V * A
  Y3 = U * (V ^ 2 * V2 - A) - V ^ 3 * U2
  Z3 = V ^ 3 * W
  return (X3, Y3, Z3)
  */

  u1.ModMul(&p2.y,&p1.z);
  u2.ModMul(&p1.y,&p2.z);
  v1.ModMul(&p2.x,&p1.z);
  v2.ModMul(&p1.x,&p2.z);
  u.ModSub(&u1,&u2);
  v.ModSub(&v1,&v2);
  w.ModMul(&p1.z,&p2.z);
  us2.ModSquare(&u);
  vs2.ModSquare(&v);
  vs3.ModMul(&vs2,&v);
  us2w.ModMul(&us2,&w);
  vs2v2.ModMul(&vs2,&v2);
  _2vs2v2.ModAdd(&vs2v2,&vs2v2);
  a.ModSub(&us2w,&vs3);
  a.ModSub(&_2vs2v2);

  r.x.ModMul(&v,&a);

  vs3u2.ModMul(&vs3,&u2);
  r.y.ModSub(&vs2v2,&a);
  r.y.ModMul(&r.y,&u);
  r.y.ModSub(&vs3u2);

  r.z.ModMul(&vs3,&w);

  return r;
}

Point Secp256K1::DoubleDirect(Point &p) {

  Int _s;
  Int _p;
  Int a;
  Point r;
  r.z.SetInt32(1);

  _s.ModMul(&p.x,&p.x);
  _p.ModAdd(&_s,&_s);
  _p.ModAdd(&_s);

  a.ModAdd(&p.y,&p.y);
  a.ModInv();
  _s.ModMul(&_p,&a);     // s = (3*pow2(p.x))*inverse(2*p.y);

  _p.ModMul(&_s,&_s);
  a.ModAdd(&p.x,&p.x);
  a.ModNeg();
  r.x.ModAdd(&a,&_p);    // rx = pow2(s) + neg(2*p.x);

  a.ModSub(&r.x,&p.x);

  _p.ModMul(&a,&_s);
  r.y.ModAdd(&_p,&p.y);
  r.y.ModNeg();           // ry = neg(p.y + s*(ret.x+neg(p.x)));  

  return r;
}

Point Secp256K1::Double(Point &p) {


  /*
  if (Y == 0)
    return POINT_AT_INFINITY
    W = a * Z ^ 2 + 3 * X ^ 2
    S = Y * Z
    B = X * Y*S
    H = W ^ 2 - 8 * B
    X' = 2*H*S
    Y' = W*(4*B - H) - 8*Y^2*S^2
    Z' = 8*S^3
    return (X', Y', Z')
  */

  Int z2;
  Int x2;
  Int _3x2;
  Int w;
  Int s;
  Int s2;
  Int b;
  Int _8b;
  Int _8y2s2;
  Int y2;
  Int h;
  Point r;

  z2.ModSquare(&p.z);
  z2.SetInt32(0); // a=0
  x2.ModSquare(&p.x);
  _3x2.ModAdd(&x2,&x2);
  _3x2.ModAdd(&x2);
  w.ModAdd(&z2,&_3x2);
  s.ModMul(&p.y,&p.z);
  b.ModMul(&p.y,&s);
  b.ModMul(&p.x);
  h.ModSquare(&w);
  _8b.ModAdd(&b,&b);
  _8b.ModDouble();
  _8b.ModDouble();
  h.ModSub(&_8b);

  r.x.ModMul(&h,&s);
  r.x.ModAdd(&r.x);

  s2.ModSquare(&s);
  y2.ModSquare(&p.y);
  _8y2s2.ModMul(&y2,&s2);
  _8y2s2.ModDouble();
  _8y2s2.ModDouble();
  _8y2s2.ModDouble();

  r.y.ModAdd(&b,&b);
  r.y.ModAdd(&r.y,&r.y);
  r.y.ModSub(&h);
  r.y.ModMul(&w);
  r.y.ModSub(&_8y2s2);

  r.z.ModMul(&s2,&s);
  r.z.ModDouble();
  r.z.ModDouble();
  r.z.ModDouble();

  return r;
}

Int Secp256K1::GetY(Int x,bool isEven) {

  Int _s;
  Int _p;

  _s.ModSquare(&x);
  _p.ModMul(&_s,&x);
  _p.ModAdd(7);
  _p.ModSqrt();

  if(!_p.IsEven() && isEven) {
    _p.ModNeg();
  }
  else if(_p.IsEven() && !isEven) {
    _p.ModNeg();
  }

  return _p;

}

int Secp256K1::EC(Point &p) {

  Int _s;
  Int _p;

  _s.ModSquare(&p.x);
  _p.ModMul(&_s,&p.x);
  _p.ModAdd(7);
  _s.ModMul(&p.y,&p.y);
  _s.ModSub(&_p);

  return _s.IsZero(); // ( ((pow2(y) - (pow3(x) + 7)) % P) == 0 );

}
