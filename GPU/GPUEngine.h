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

#ifndef GPUENGINEH
#define GPUENGINEH

#include <vector>
#include "../SECP256k1.h"

// Number of key per thread (must be a multiple of GRP_SIZE) per kernel call
#define STEP_SIZE 1024

// Number of thread per block
#define NB_TRHEAD_PER_GROUP 128

// Maximum number of 16bit prefix found per kernel 
// Avg = (nbThread*STEP_SIZE*nbPrefix16)/65536
#define MAX_FOUND 16384
#define ITEM_SIZE 28
#define ITEM_SIZE32 (ITEM_SIZE/4)
#define OUTPUT_SIZE (MAX_FOUND*ITEM_SIZE+4)
#define _64K 65536

typedef uint16_t prefix_t;
typedef uint32_t prefixl_t;

typedef struct {
  uint32_t thId;
  int32_t  incr;
  uint8_t  *hash;
} ITEM;

// Second level lookup
typedef struct {
  prefix_t sPrefix;
  std::vector<prefixl_t> lPrefixes;
} LPREFIX;

class GPUEngine {

public:

  GPUEngine(int nbThreadGroup,int gpuId); 
  ~GPUEngine();
  void SetPrefix(std::vector<prefix_t> prefixes);
  void SetPrefix(std::vector<LPREFIX> prefixes,uint32_t totalPrefix);
  bool SetKeys(Point *p);
  void SetSearchMode(bool compressed);
  bool Launch(std::vector<ITEM> &prefixFound,bool spinWait=false);
  int GetNbThread();
  int GetGroupSize();

  bool Check(Secp256K1 &secp);
  std::string deviceName;

  static void PrintCudaInfo();
  static void GenerateCode(Secp256K1 &secp, int size);

private:

  bool callKernel();
  static void ComputeIndex(std::vector<int> &s, int depth, int n);
  static void Browse(FILE *f,int depth, int max, int s);
  bool CheckHash(uint8_t *h, std::vector<ITEM>& found);

  int nbThread;
  prefix_t *inputPrefix;
  prefix_t *inputPrefixPinned;
  uint32_t *inputPrefixLookUp;
  uint32_t *inputPrefixLookUpPinned;
  uint64_t *inputKey;
  uint64_t *inputKeyPinned;
  uint32_t *outputPrefix;
  uint32_t *outputPrefixPinned;
  bool initialised;
  bool searchComp;
  bool littleEndian;
  bool lostWarning;
};

#endif // GPUENGINEH
