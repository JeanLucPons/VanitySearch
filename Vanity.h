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

class VanitySearch;

typedef struct {

  VanitySearch *obj;
  int threadId;
  bool isRunning;
  int gridSize;
  int gpuId;

} TH_PARAM;

class VanitySearch {

public:

  VanitySearch(Secp256K1 &secp, std::string prefix, std::string seed, bool compressed, 
               bool useGpu,bool stop,std::string outputFile, bool useSSE);
  void Search(int nbThread,std::vector<int> gpuId,std::vector<int> gridSize);
  void FindKeyCPU(TH_PARAM *p);
  void FindKeyGPU(TH_PARAM *p);

private:

  std::string GetHex(std::vector<unsigned char> &buffer);
  std::string GetExpectedTime(double keyRate, double keyCount);
  bool checkAddr(std::string &addr, Int &key, uint64_t incr);
  void output(std::string addr, std::string pAddr, std::string pAddrHex, std::string chkAddr, std::string chkAddrC);
  bool isAlive(TH_PARAM *p);
  uint64_t getGPUCount();
  uint64_t getCPUCount();

  Secp256K1 secp;
  double _difficulty;
  std::string vanityPrefix;
  prefix_t sPrefix;
  Int startKey;
  uint64_t counters[256];
  double startTime;
  bool searchComp;
  bool useGpu;
  bool stopWhenFound;
  bool endOfSearch;
  int nbCPUThread;
  int nbGPUThread;
  int nbFoundKey;
  std::string outputFile;
  bool useSSE;

#ifdef WIN64
  HANDLE ghMutex;
#else
  pthread_mutex_t  ghMutex;
#endif

};

#endif // VANITYH
