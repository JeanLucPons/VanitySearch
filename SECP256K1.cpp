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
#include "Bech32.h"
#include <string.h>

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

  Int::InitK1(&order);

  // Compute Generator table
  Point N(G);
  for(int i = 0; i < 32; i++) {
    GTable[i * 256] = N;
    N = DoubleDirect(N);
    for (int j = 1; j < 255; j++) {
      GTable[i * 256 + j] = N;
      N = AddDirect(N, GTable[i * 256]);
    }
    GTable[i * 256 + 255] = N; // Dummy point for check function
  }

}

Secp256K1::~Secp256K1() {
}

void PrintResult(bool ok) {
  if(ok) {
    printf("OK\n");
  }
  else {
    printf("Failed !\n");
  }
}

void CheckAddress(Secp256K1 *T,std::string address,std::string privKeyStr) {

  bool isCompressed;
  int type;

  Int privKey = T->DecodePrivateKey((char *)privKeyStr.c_str(),&isCompressed);
  Point pub = T->ComputePublicKey(&privKey);

  switch (address.data()[0]) {
  case '1':
    type = P2PKH; break;
  case '3':
    type = P2SH; break;
  case 'b':
  case 'B':
    type = BECH32; break;
  default:
    printf("Failed ! \n%s Address format not supported\n", address.c_str());
    return;
  }

  std::string calcAddress = T->GetAddress(type, isCompressed, pub);

  printf("Adress : %s ", address.c_str());

  if (address == calcAddress) {
    printf("OK!\n");
    return;
  }

  printf("Failed ! \n %s\n", calcAddress.c_str());

}

void Secp256K1::Check() {

  printf("Check Generator :");

  bool ok = true;
  int i = 0;
  while(i < 256*32 && EC(GTable[i])) {
    i++;
  }
  PrintResult(i == 256*32);

  printf("Check Double :");
  Point Pt(G);
  Point R1;
  Point R2;
  Point R3;
  R1 = Double(G);
  R1.Reduce();
  PrintResult(EC(R1));

  printf("Check Add :");
  R2 = Add(G,R1);
  R3 = Add(R1,R2);
  R3.Reduce();
  PrintResult(EC(R3));

  printf("Check GenKey :");
  Int privKey;
  privKey.SetBase16("46b9e861b63d3509c88b7817275a30d22d62c8cd8fa6486ddee35ef0d8e0495f");
  Point pub = ComputePublicKey(&privKey);
  Point expectedPubKey;
  expectedPubKey.x.SetBase16("2500e7f3fbddf2842903f544ddc87494ce95029ace4e257d54ba77f2bc1f3a88");
  expectedPubKey.y.SetBase16("37a9461c4f1c57fecc499753381e772a128a5820a924a2fa05162eb662987a9f");
  expectedPubKey.z.SetInt32(1);

  PrintResult(pub.equals(expectedPubKey));

  CheckAddress(this,"15t3Nt1zyMETkHbjJTTshxLnqPzQvAtdCe","5HqoeNmaz17FwZRqn7kCBP1FyJKSe4tt42XZB7426EJ2MVWDeqk");
  CheckAddress(this,"1BoatSLRHtKNngkdXEeobR76b53LETtpyT","5J4XJRyLVgzbXEgh8VNi4qovLzxRftzMd8a18KkdXv4EqAwX3tS");
  CheckAddress(this,"1Test6BNjSJC5qwYXsjwKVLvz7DpfLehy","5HytzR8p5hp8Cfd8jsVFnwMNXMsEW1sssFxMQYqEUjGZN72iLJ2");
  CheckAddress(this,"16S5PAsGZ8VFM1CRGGLqm37XHrp46f6CTn","KxMUSkFhEzt2eJHscv2vNSTnnV2cgAXgL4WDQBTx7Ubd9TZmACAz");
  CheckAddress(this,"1Tst2RwMxZn9cYY5mQhCdJic3JJrK7Fq7","L1vamTpSeK9CgynRpSJZeqvUXf6dJa25sfjb2uvtnhj65R5TymgF");
  CheckAddress(this,"3CyQYcByvcWK8BkYJabBS82yDLNWt6rWSx","KxMUSkFhEzt2eJHscv2vNSTnnV2cgAXgL4WDQBTx7Ubd9TZmACAz");
  CheckAddress(this,"31to1KQe67YjoDfYnwFJThsGeQcFhVDM5Q","KxV2Tx5jeeqLHZ1V9ufNv1doTZBZuAc5eY24e6b27GTkDhYwVad7");
  CheckAddress(this,"bc1q6tqytpg06uhmtnhn9s4f35gkt8yya5a24dptmn","L2wAVD273GwAxGuEDHvrCqPfuWg5wWLZWy6H3hjsmhCvNVuCERAQ");

  // 1ViViGLEawN27xRzGrEhhYPQrZiTKvKLo
  pub.x.SetBase16(/*04*/"75249c39f38baa6bf20ab472191292349426dc3652382cdc45f65695946653dc");
  pub.y.SetBase16("978b2659122fe1df1be132167f27b74e5d4a2f3ecbbbd0b3fbcc2f4983518674");
  printf("Check Calc PubKey (full) %s :",GetAddress(P2PKH, false,pub).c_str());
  PrintResult(EC(pub));

  // 385cR5DM96n1HvBDMzLHPYcw89fZAXULJP
  pub.x.SetBase16(/*03*/"c931af9f331b7a9eb2737667880dacb91428906fbffad0173819a873172d21c4");
  pub.y = GetY(pub.x,false);
  printf("Check Calc PubKey (even) %s:",GetAddress(P2SH, true, pub).c_str());
  PrintResult(EC(pub));

  // 18aPiLmTow7Xgu96msrDYvSSWweCvB9oBA
  pub.x.SetBase16(/*03*/"3bf3d80f868fa33c6353012cb427e98b080452f19b5c1149ea2acfe4b7599739");
  pub.y = GetY(pub.x,false);
  printf("Check Calc PubKey (odd) %s:",GetAddress(P2PKH, true, pub).c_str());
  PrintResult(EC(pub));

}


Point Secp256K1::ComputePublicKey(Int *privKey) {

  int i = 0;
  uint8_t b;
  Point Q;
  Q.Clear();

  // Search first significant byte
  for (i = 0; i < 32; i++) {
    b = privKey->GetByte(i);
    if(b)
      break;
  }
  Q = GTable[256 * i + (b-1)];
  i++;

  for(; i < 32; i++) {
    b = privKey->GetByte(i);
    if(b)
      Q = Add2(Q, GTable[256 * i + (b-1)]);
  }

  Q.Reduce();
  return Q;

}

Point Secp256K1::NextKey(Point &key) {
  // Input key must be reduced and different from G
  // in order to use AddDirect
  return AddDirect(key,G);
}

Int Secp256K1::DecodePrivateKey(char *key,bool *compressed) {

  Int ret;
  ret.SetInt32(0);
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

    // Compute checksum
    unsigned char c[4];
    sha256_checksum(privKey.data(), 33, c);

    if( c[0]!=privKey[33] || c[1]!=privKey[34] ||
        c[2]!=privKey[35] || c[3]!=privKey[36] ) {
      printf("Warning, Invalid private key checksum !\n");
    }

    *compressed = false;
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

    // Compute checksum
    unsigned char c[4];
    sha256_checksum(privKey.data(), 34, c);

    if( c[0]!=privKey[34] || c[1]!=privKey[35] ||
        c[2]!=privKey[36] || c[3]!=privKey[37] ) {
      printf("Warning, Invalid private key checksum !\n");
    }

    *compressed = true;
    return ret;

  }

  printf("Invalid private key, not starting with 5,K or L !\n");
  ret.SetInt32(-1);
  return ret;

}

#define KEYBUFFCOMP(buff,p) \
(buff)[0] = ((p).x.bits[7] >> 8) | ((uint32_t)(0x2 + (p).y.IsOdd()) << 24); \
(buff)[1] = ((p).x.bits[6] >> 8) | ((p).x.bits[7] <<24); \
(buff)[2] = ((p).x.bits[5] >> 8) | ((p).x.bits[6] <<24); \
(buff)[3] = ((p).x.bits[4] >> 8) | ((p).x.bits[5] <<24); \
(buff)[4] = ((p).x.bits[3] >> 8) | ((p).x.bits[4] <<24); \
(buff)[5] = ((p).x.bits[2] >> 8) | ((p).x.bits[3] <<24); \
(buff)[6] = ((p).x.bits[1] >> 8) | ((p).x.bits[2] <<24); \
(buff)[7] = ((p).x.bits[0] >> 8) | ((p).x.bits[1] <<24); \
(buff)[8] = 0x00800000 | ((p).x.bits[0] <<24); \
(buff)[9] = 0; \
(buff)[10] = 0; \
(buff)[11] = 0; \
(buff)[12] = 0; \
(buff)[13] = 0; \
(buff)[14] = 0; \
(buff)[15] = 0x108;

#define KEYBUFFUNCOMP(buff,p) \
(buff)[0] = ((p).x.bits[7] >> 8) | 0x04000000; \
(buff)[1] = ((p).x.bits[6] >> 8) | ((p).x.bits[7] <<24); \
(buff)[2] = ((p).x.bits[5] >> 8) | ((p).x.bits[6] <<24); \
(buff)[3] = ((p).x.bits[4] >> 8) | ((p).x.bits[5] <<24); \
(buff)[4] = ((p).x.bits[3] >> 8) | ((p).x.bits[4] <<24); \
(buff)[5] = ((p).x.bits[2] >> 8) | ((p).x.bits[3] <<24); \
(buff)[6] = ((p).x.bits[1] >> 8) | ((p).x.bits[2] <<24); \
(buff)[7] = ((p).x.bits[0] >> 8) | ((p).x.bits[1] <<24); \
(buff)[8] = ((p).y.bits[7] >> 8) | ((p).x.bits[0] <<24); \
(buff)[9] = ((p).y.bits[6] >> 8) | ((p).y.bits[7] <<24); \
(buff)[10] = ((p).y.bits[5] >> 8) | ((p).y.bits[6] <<24); \
(buff)[11] = ((p).y.bits[4] >> 8) | ((p).y.bits[5] <<24); \
(buff)[12] = ((p).y.bits[3] >> 8) | ((p).y.bits[4] <<24); \
(buff)[13] = ((p).y.bits[2] >> 8) | ((p).y.bits[3] <<24); \
(buff)[14] = ((p).y.bits[1] >> 8) | ((p).y.bits[2] <<24); \
(buff)[15] = ((p).y.bits[0] >> 8) | ((p).y.bits[1] <<24); \
(buff)[16] = 0x00800000 | ((p).y.bits[0] <<24); \
(buff)[17] = 0; \
(buff)[18] = 0; \
(buff)[19] = 0; \
(buff)[20] = 0; \
(buff)[21] = 0; \
(buff)[22] = 0; \
(buff)[23] = 0; \
(buff)[24] = 0; \
(buff)[25] = 0; \
(buff)[26] = 0; \
(buff)[27] = 0; \
(buff)[28] = 0; \
(buff)[29] = 0; \
(buff)[30] = 0; \
(buff)[31] = 0x208;

#define KEYBUFFSCRIPT(buff,h) \
(buff)[0] = 0x00140000 | (uint32_t)h[0] << 8 | (uint32_t)h[1]; \
(buff)[1] = (uint32_t)h[2] << 24 | (uint32_t)h[3] << 16 | (uint32_t)h[4] << 8 | (uint32_t)h[5];\
(buff)[2] = (uint32_t)h[6] << 24 | (uint32_t)h[7] << 16 | (uint32_t)h[8] << 8 | (uint32_t)h[9];\
(buff)[3] = (uint32_t)h[10] << 24 | (uint32_t)h[11] << 16 | (uint32_t)h[12] << 8 | (uint32_t)h[13];\
(buff)[4] = (uint32_t)h[14] << 24 | (uint32_t)h[15] << 16 | (uint32_t)h[16] << 8 | (uint32_t)h[17];\
(buff)[5] = (uint32_t)h[18] << 24 | (uint32_t)h[19] << 16 | 0x8000; \
(buff)[6] = 0; \
(buff)[7] = 0; \
(buff)[8] = 0; \
(buff)[9] = 0; \
(buff)[10] = 0; \
(buff)[11] = 0; \
(buff)[12] = 0; \
(buff)[13] = 0; \
(buff)[14] = 0; \
(buff)[15] = 0xB0;

void Secp256K1::GetHash160(int type,bool compressed,
  Point &k0,Point &k1,Point &k2,Point &k3,
  uint8_t *h0,uint8_t *h1,uint8_t *h2,uint8_t *h3) {

#ifdef WIN64
  __declspec(align(16)) unsigned char sh0[64];
  __declspec(align(16)) unsigned char sh1[64];
  __declspec(align(16)) unsigned char sh2[64];
  __declspec(align(16)) unsigned char sh3[64];
#else
  unsigned char sh0[64] __attribute__((aligned(16)));
  unsigned char sh1[64] __attribute__((aligned(16)));
  unsigned char sh2[64] __attribute__((aligned(16)));
  unsigned char sh3[64] __attribute__((aligned(16)));
#endif

  switch (type) {

  case P2PKH:
  case BECH32:
  {

    if (!compressed) {

      uint32_t b0[32];
      uint32_t b1[32];
      uint32_t b2[32];
      uint32_t b3[32];

      KEYBUFFUNCOMP(b0, k0);
      KEYBUFFUNCOMP(b1, k1);
      KEYBUFFUNCOMP(b2, k2);
      KEYBUFFUNCOMP(b3, k3);

      sha256sse_2B(b0, b1, b2, b3, sh0, sh1, sh2, sh3);
      ripemd160sse_32(sh0, sh1, sh2, sh3, h0, h1, h2, h3);

    } else {

      uint32_t b0[16];
      uint32_t b1[16];
      uint32_t b2[16];
      uint32_t b3[16];

      KEYBUFFCOMP(b0, k0);
      KEYBUFFCOMP(b1, k1);
      KEYBUFFCOMP(b2, k2);
      KEYBUFFCOMP(b3, k3);

      sha256sse_1B(b0, b1, b2, b3, sh0, sh1, sh2, sh3);
      ripemd160sse_32(sh0, sh1, sh2, sh3, h0, h1, h2, h3);

    }

  }
  break;

  case P2SH:
  {

    unsigned char kh0[20];
    unsigned char kh1[20];
    unsigned char kh2[20];
    unsigned char kh3[20];

    GetHash160(P2PKH,compressed,k0,k1,k2,k3,kh0,kh1,kh2,kh3);

    // Redeem Script (1 to 1 P2SH)
    uint32_t b0[16];
    uint32_t b1[16];
    uint32_t b2[16];
    uint32_t b3[16];

    KEYBUFFSCRIPT(b0, kh0);
    KEYBUFFSCRIPT(b1, kh1);
    KEYBUFFSCRIPT(b2, kh2);
    KEYBUFFSCRIPT(b3, kh3);

    sha256sse_1B(b0, b1, b2, b3, sh0, sh1, sh2, sh3);
    ripemd160sse_32(sh0, sh1, sh2, sh3, h0, h1, h2, h3);

  }
  break;

  }

}

uint8_t Secp256K1::GetByte(std::string &str, int idx) {

  char tmp[3];
  int  val;

  tmp[0] = str.data()[2 * idx];
  tmp[1] = str.data()[2 * idx + 1];
  tmp[2] = 0;

  if (sscanf(tmp, "%X", &val) != 1) {
    printf("ParsePublicKeyHex: Error invalid public key specified (unexpected hexadecimal digit)\n");
    exit(-1);
  }

  return (uint8_t)val;

}

Point Secp256K1::ParsePublicKeyHex(std::string str,bool &isCompressed) {

  Point ret;
  ret.Clear();

  if (str.length() < 2) {
    printf("ParsePublicKeyHex: Error invalid public key specified (66 or 130 character length)\n");
    exit(-1);
  }

  uint8_t type = GetByte(str, 0);

  switch (type) {

    case 0x02:
      if (str.length() != 66) {
        printf("ParsePublicKeyHex: Error invalid public key specified (66 character length)\n");
        exit(-1);
      }
      for (int i = 0; i < 32; i++)
        ret.x.SetByte(31 - i, GetByte(str, i + 1));
      ret.y = GetY(ret.x, true);
      isCompressed = true;
      break;

    case 0x03:
      if (str.length() != 66) {
        printf("ParsePublicKeyHex: Error invalid public key specified (66 character length)\n");
        exit(-1);
      }
      for (int i = 0; i < 32; i++)
        ret.x.SetByte(31 - i, GetByte(str, i + 1));
      ret.y = GetY(ret.x, false);
      isCompressed = true;
      break;

    case 0x04:
      if (str.length() != 130) {
        printf("ParsePublicKeyHex: Error invalid public key specified (130 character length)\n");
        exit(-1);
      }
      for (int i = 0; i < 32; i++)
        ret.x.SetByte(31 - i, GetByte(str, i + 1));
      for (int i = 0; i < 32; i++)
        ret.y.SetByte(31 - i, GetByte(str, i + 33));
      isCompressed = false;
      break;

    default:
      printf("ParsePublicKeyHex: Error invalid public key specified (Unexpected prefix (only 02,03 or 04 allowed)\n");
      exit(-1);
  }

  ret.z.SetInt32(1);

  if (!EC(ret)) {
    printf("ParsePublicKeyHex: Error invalid public key specified (Not lie on elliptic curve)\n");
    exit(-1);
  }

  return ret;

}

std::string Secp256K1::GetPublicKeyHex(bool compressed, Point &pubKey) {

  unsigned char publicKeyBytes[128];
  char tmp[3];
  std::string ret;

  if (!compressed) {

    // Full public key
    publicKeyBytes[0] = 0x4;
    pubKey.x.Get32Bytes(publicKeyBytes + 1);
    pubKey.y.Get32Bytes(publicKeyBytes + 33);

    for (int i = 0; i < 65; i++) {
      sprintf(tmp, "%02X", (int)publicKeyBytes[i]);
      ret.append(tmp);
    }

  } else {

    // Compressed public key
    publicKeyBytes[0] = pubKey.y.IsEven() ? 0x2 : 0x3;
    pubKey.x.Get32Bytes(publicKeyBytes + 1);

    for (int i = 0; i < 33; i++) {
      sprintf(tmp, "%02X", (int)publicKeyBytes[i]);
      ret.append(tmp);
    }

  }

  return ret;

}

void Secp256K1::GetHash160(int type, bool compressed, Point &pubKey, unsigned char *hash) {

  unsigned char shapk[64];

  switch (type) {

  case P2PKH:
  case BECH32:
  {
    unsigned char publicKeyBytes[128];

    if (!compressed) {

      // Full public key
      publicKeyBytes[0] = 0x4;
      pubKey.x.Get32Bytes(publicKeyBytes + 1);
      pubKey.y.Get32Bytes(publicKeyBytes + 33);
      sha256_65(publicKeyBytes, shapk);

    } else {

      // Compressed public key
      publicKeyBytes[0] = pubKey.y.IsEven() ? 0x2 : 0x3;
      pubKey.x.Get32Bytes(publicKeyBytes + 1);
      sha256_33(publicKeyBytes, shapk);

    }

    ripemd160_32(shapk, hash);
  }
  break;

  case P2SH:
  {

    // Redeem Script (1 to 1 P2SH)
    unsigned char script[64];

    script[0] = 0x00;  // OP_0
    script[1] = 0x14;  // PUSH 20 bytes
    GetHash160(P2PKH, compressed, pubKey, script + 2);

    sha256(script, 22, shapk);
    ripemd160_32(shapk, hash);

  }
  break;

  }

}

std::string Secp256K1::GetPrivAddress(bool compressed,Int &privKey) {

  unsigned char address[38];

  address[0] = 0x80; // Mainnet
  privKey.Get32Bytes(address + 1);

  if( compressed ) {

    // compressed suffix
    address[33] = 1;
    sha256_checksum(address, 34, address + 34);
    return EncodeBase58(address,address + 38);

  } else {

    // Compute checksum
    sha256_checksum(address, 33, address + 33);
    return EncodeBase58(address,address + 37);

  }

}

#define CHECKSUM(buff,A) \
(buff)[0] = (uint32_t)A[0] << 24 | (uint32_t)A[1] << 16 | (uint32_t)A[2] << 8 | (uint32_t)A[3];\
(buff)[1] = (uint32_t)A[4] << 24 | (uint32_t)A[5] << 16 | (uint32_t)A[6] << 8 | (uint32_t)A[7];\
(buff)[2] = (uint32_t)A[8] << 24 | (uint32_t)A[9] << 16 | (uint32_t)A[10] << 8 | (uint32_t)A[11];\
(buff)[3] = (uint32_t)A[12] << 24 | (uint32_t)A[13] << 16 | (uint32_t)A[14] << 8 | (uint32_t)A[15];\
(buff)[4] = (uint32_t)A[16] << 24 | (uint32_t)A[17] << 16 | (uint32_t)A[18] << 8 | (uint32_t)A[19];\
(buff)[5] = (uint32_t)A[20] << 24 | 0x800000;\
(buff)[6] = 0; \
(buff)[7] = 0; \
(buff)[8] = 0; \
(buff)[9] = 0; \
(buff)[10] = 0; \
(buff)[11] = 0; \
(buff)[12] = 0; \
(buff)[13] = 0; \
(buff)[14] = 0; \
(buff)[15] = 0xA8;

std::vector<std::string> Secp256K1::GetAddress(int type, bool compressed, unsigned char *h1, unsigned char *h2, unsigned char *h3, unsigned char *h4) {

  std::vector<std::string> ret;

  unsigned char add1[25];
  unsigned char add2[25];
  unsigned char add3[25];
  unsigned char add4[25];
  uint32_t b1[16];
  uint32_t b2[16];
  uint32_t b3[16];
  uint32_t b4[16];

  switch (type) {

  case P2PKH:
    add1[0] = 0x00;
    add2[0] = 0x00;
    add3[0] = 0x00;
    add4[0] = 0x00;
    break;

  case P2SH:
    add1[0] = 0x05;
    add2[0] = 0x05;
    add3[0] = 0x05;
    add4[0] = 0x05;
    break;

  case BECH32:
  {
    char output[128];
    segwit_addr_encode(output, "bc", 0, h1, 20);
    ret.push_back(std::string(output));
    segwit_addr_encode(output, "bc", 0, h2, 20);
    ret.push_back(std::string(output));
    segwit_addr_encode(output, "bc", 0, h3, 20);
    ret.push_back(std::string(output));
    segwit_addr_encode(output, "bc", 0, h4, 20);
    ret.push_back(std::string(output));
    return ret;
  }
  break;
  }

  memcpy(add1 + 1, h1, 20);
  memcpy(add2 + 1, h2, 20);
  memcpy(add3 + 1, h3, 20);
  memcpy(add4 + 1, h4, 20);
  CHECKSUM(b1, add1);
  CHECKSUM(b2, add2);
  CHECKSUM(b3, add3);
  CHECKSUM(b4, add4);
  sha256sse_checksum(b1,b2,b3,b4,add1 + 21, add2 + 21, add3 + 21, add4 + 21);

  // Base58
  ret.push_back(EncodeBase58(add1, add1 + 25));
  ret.push_back(EncodeBase58(add2, add2 + 25));
  ret.push_back(EncodeBase58(add3, add3 + 25));
  ret.push_back(EncodeBase58(add4, add4 + 25));

  return ret;

}

std::string Secp256K1::GetAddress(int type, bool compressed,unsigned char *hash160) {

  unsigned char address[25];
  switch(type) {

    case P2PKH:
      address[0] = 0x00;
      break;

    case P2SH:
      address[0] = 0x05;
      break;

    case BECH32:
    {
      char output[128];
      segwit_addr_encode(output, "bc", 0, hash160, 20);
      return std::string(output);
    }
    break;
  }
  memcpy(address + 1, hash160,20);
  sha256_checksum(address,21,address+21);

  // Base58
  return EncodeBase58(address, address + 25);

}

std::string Secp256K1::GetAddress(int type, bool compressed, Point &pubKey) {

  unsigned char address[25];

  switch (type) {

  case P2PKH:
    address[0] = 0x00;
    break;

  case BECH32:
  {
    if (!compressed) {
      return " BECH32: Only compressed key ";
    }
    char output[128];
    uint8_t h160[20];
    GetHash160(type, compressed, pubKey, h160);
    segwit_addr_encode(output,"bc",0,h160,20);
    return std::string(output);
  }
  break;

  case P2SH:
    if (!compressed) {
      return " P2SH: Only compressed key ";
    }
    address[0] = 0x05;
    break;
  }

  GetHash160(type,compressed,pubKey, address + 1);
  sha256_checksum(address, 21, address + 21);

  // Base58
  return EncodeBase58(address, address + 25);

}

bool Secp256K1::CheckPudAddress(std::string address) {

  std::vector<unsigned char> pubKey;
  DecodeBase58(address,pubKey);

  if(pubKey.size()!=25)
    return false;

  // Check checksum
  unsigned char chk[4];
  sha256_checksum(pubKey.data(), 21, chk);

  return  (pubKey[21] == chk[0]) &&
          (pubKey[22] == chk[1]) &&
          (pubKey[23] == chk[2]) &&
          (pubKey[24] == chk[3]);

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
  _s.ModMulK1(&dy,&dx);     // s = (p2.y-p1.y)*inverse(p2.x-p1.x);

  _p.ModSquareK1(&_s);       // _p = pow2(s)

  r.x.ModSub(&_p,&p1.x);
  r.x.ModSub(&p2.x);       // rx = pow2(s) - p1.x - p2.x;

  r.y.ModSub(&p2.x,&r.x);
  r.y.ModMulK1(&_s);
  r.y.ModSub(&p2.y);       // ry = - p2.y - s*(ret.x-p2.x);

  return r;

}

Point Secp256K1::Add2(Point &p1, Point &p2) {

  // P2.z = 1

  Int u;
  Int v;
  Int u1;
  Int v1;
  Int vs2;
  Int vs3;
  Int us2;
  Int a;
  Int us2w;
  Int vs2v2;
  Int vs3u2;
  Int _2vs2v2;
  Point r;

  u1.ModMulK1(&p2.y, &p1.z);
  v1.ModMulK1(&p2.x, &p1.z);
  u.ModSub(&u1, &p1.y);
  v.ModSub(&v1, &p1.x);
  us2.ModSquareK1(&u);
  vs2.ModSquareK1(&v);
  vs3.ModMulK1(&vs2, &v);
  us2w.ModMulK1(&us2, &p1.z);
  vs2v2.ModMulK1(&vs2, &p1.x);
  _2vs2v2.ModAdd(&vs2v2, &vs2v2);
  a.ModSub(&us2w, &vs3);
  a.ModSub(&_2vs2v2);

  r.x.ModMulK1(&v, &a);

  vs3u2.ModMulK1(&vs3, &p1.y);
  r.y.ModSub(&vs2v2, &a);
  r.y.ModMulK1(&r.y, &u);
  r.y.ModSub(&vs3u2);

  r.z.ModMulK1(&vs3, &p1.z);

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

  u1.ModMulK1(&p2.y,&p1.z);
  u2.ModMulK1(&p1.y,&p2.z);
  v1.ModMulK1(&p2.x,&p1.z);
  v2.ModMulK1(&p1.x,&p2.z);
  u.ModSub(&u1,&u2);
  v.ModSub(&v1,&v2);
  w.ModMulK1(&p1.z,&p2.z);
  us2.ModSquareK1(&u);
  vs2.ModSquareK1(&v);
  vs3.ModMulK1(&vs2,&v);
  us2w.ModMulK1(&us2,&w);
  vs2v2.ModMulK1(&vs2,&v2);
  _2vs2v2.ModAdd(&vs2v2,&vs2v2);
  a.ModSub(&us2w,&vs3);
  a.ModSub(&_2vs2v2);

  r.x.ModMulK1(&v,&a);

  vs3u2.ModMulK1(&vs3,&u2);
  r.y.ModSub(&vs2v2,&a);
  r.y.ModMulK1(&r.y,&u);
  r.y.ModSub(&vs3u2);

  r.z.ModMulK1(&vs3,&w);

  return r;
}

Point Secp256K1::DoubleDirect(Point &p) {

  Int _s;
  Int _p;
  Int a;
  Point r;
  r.z.SetInt32(1);

  _s.ModMulK1(&p.x,&p.x);
  _p.ModAdd(&_s,&_s);
  _p.ModAdd(&_s);

  a.ModAdd(&p.y,&p.y);
  a.ModInv();
  _s.ModMulK1(&_p,&a);     // s = (3*pow2(p.x))*inverse(2*p.y);

  _p.ModMulK1(&_s,&_s);
  a.ModAdd(&p.x,&p.x);
  a.ModNeg();
  r.x.ModAdd(&a,&_p);    // rx = pow2(s) + neg(2*p.x);

  a.ModSub(&r.x,&p.x);

  _p.ModMulK1(&a,&_s);
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

  z2.ModSquareK1(&p.z);
  z2.SetInt32(0); // a=0
  x2.ModSquareK1(&p.x);
  _3x2.ModAdd(&x2,&x2);
  _3x2.ModAdd(&x2);
  w.ModAdd(&z2,&_3x2);
  s.ModMulK1(&p.y,&p.z);
  b.ModMulK1(&p.y,&s);
  b.ModMulK1(&p.x);
  h.ModSquareK1(&w);
  _8b.ModAdd(&b,&b);
  _8b.ModDouble();
  _8b.ModDouble();
  h.ModSub(&_8b);

  r.x.ModMulK1(&h,&s);
  r.x.ModAdd(&r.x);

  s2.ModSquareK1(&s);
  y2.ModSquareK1(&p.y);
  _8y2s2.ModMulK1(&y2,&s2);
  _8y2s2.ModDouble();
  _8y2s2.ModDouble();
  _8y2s2.ModDouble();

  r.y.ModAdd(&b,&b);
  r.y.ModAdd(&r.y,&r.y);
  r.y.ModSub(&h);
  r.y.ModMulK1(&w);
  r.y.ModSub(&_8y2s2);

  r.z.ModMulK1(&s2,&s);
  r.z.ModDouble();
  r.z.ModDouble();
  r.z.ModDouble();

  return r;
}

Int Secp256K1::GetY(Int x,bool isEven) {

  Int _s;
  Int _p;

  _s.ModSquareK1(&x);
  _p.ModMulK1(&_s,&x);
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

bool Secp256K1::EC(Point &p) {

  Int _s;
  Int _p;

  _s.ModSquareK1(&p.x);
  _p.ModMulK1(&_s,&p.x);
  _p.ModAdd(7);
  _s.ModMulK1(&p.y,&p.y);
  _s.ModSub(&_p);

  return _s.IsZero(); // ( ((pow2(y) - (pow3(x) + 7)) % P) == 0 );

}
