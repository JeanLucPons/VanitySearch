#ifndef SECP256K1H
#define SECP256K1H

#include "Point.h"
#include <string>
#include <vector>

class Secp256K1 {

public:

  Secp256K1();
  ~Secp256K1();
  void Init();
  Point ComputePublicKey(Int *privKey);
  Point NextKey(Point &key);
  void Secp256K1::Check();
  int  EC(Point &p);
  void GetHash160(bool compressed,
    Point &k0, Point &k1, Point &k2, Point &k3,
    uint8_t *h0, uint8_t *h1, uint8_t *h2, uint8_t *h3);
  void GetHash160(Point &pubKey,bool compressed,unsigned char *hash);
  std::string GetAddress(Point &pubKey,bool compressed);
  std::string GetAddress(unsigned char *hash20, bool compressed);
  std::string GetPrivAddress(Int &privKey);

  static Int DecodePrivateKey(char *key);

  Point Add(Point &p1, Point &p2);
  Point AddDirect(Point &p1, Point &p2);
  Point Double(Point &p);
  Point DoubleDirect(Point &p);

  Point G;                 // Generator
  Int   order;             // Curve order

private:

  Int GetY(Int x, bool isEven);
  Point GTable[256];       // Generator table

};

#endif // SECP256K1H
