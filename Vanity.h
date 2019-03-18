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

#ifndef VANITYH
#define VANITYH

#include <string>
#include <vector>
#include "SECP256k1.h"
#include "GPU/GPUEngine.h"
#ifdef WIN64
#include <Windows.h>
#endif

#define CPU_GRP_SIZE 1024

class VanitySearch;

typedef struct {

  VanitySearch *obj;
  int threadId;
  bool isRunning;
  bool hasStarted;
  int  gridSize;
  int  gpuId;

} TH_PARAM;

typedef struct {

  std::string prefix;
  prefix_t sPrefix;
  double difficulty;
  bool found;

  // For dreamer ;)
  bool isFull;
  prefixl_t lPrefix;
  uint8_t hash160[20];

} PREFIX_ITEM;

typedef struct {
  std::vector<PREFIX_ITEM> *items;
  bool found;
} PREFIX_TABLE_ITEM;

class VanitySearch {

public:

  VanitySearch(Secp256K1 &secp, std::vector<std::string> prefix, std::string seed, int searchMode, 
               bool useGpu,bool stop,std::string outputFile, bool useSSE);
  void Search(int nbThread,std::vector<int> gpuId,std::vector<int> gridSize);
  void FindKeyCPU(TH_PARAM *p);
  void FindKeyGPU(TH_PARAM *p);

private:

  std::string GetHex(std::vector<unsigned char> &buffer);
  std::string GetExpectedTime(double keyRate, double keyCount);
  void checkAddr(int prefIdx, uint8_t *hash160, Int &key, int32_t incr, int endomorphism, bool mode);
  void checkAddresses(bool compressed, Int key, int i, Point p1);
  void checkAddressesSSE(bool compressed, Int key, int i, Point p1, Point p2, Point p3, Point p4);
  void output(std::string addr, std::string pAddr, std::string pAddrHex, std::string chkAddr, std::string chkAddrC);
  bool isAlive(TH_PARAM *p);
  bool isSingularPrefix(std::string pref);
  bool hasStarted(TH_PARAM *p);
  uint64_t getGPUCount();
  uint64_t getCPUCount();
  bool initPrefix(std::string prefix, PREFIX_ITEM *it);
  void dumpPrefixes();
  double getDiffuclty();
  void updateFound();

  Secp256K1 secp;
  Int startKey;
  uint64_t counters[256];
  double startTime;
  int searchMode;
  bool useGpu;
  bool stopWhenFound;
  bool endOfSearch;
  int nbCPUThread;
  int nbGPUThread;
  int nbFoundKey;
  uint32_t nbPrefix;
  std::string outputFile;
  bool useSSE;
  bool onlyFull;
  double _difficulty;
  std::vector<PREFIX_TABLE_ITEM> prefixes;
  std::vector<prefix_t> usedPrefix;
  std::vector<LPREFIX> usedPrefixL;

  Int beta;
  Int lambda;
  Int beta2;
  Int lambda2;

#ifdef WIN64
  HANDLE ghMutex;
#else
  pthread_mutex_t  ghMutex;
#endif

};

#endif // VANITYH
