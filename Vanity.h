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

} TH_PARAM;

class VanitySearch {

public:

  VanitySearch(Secp256K1 &secp, std::string prefix, std::string seed, bool compressed, 
               bool useGpu, int gpuId, bool stop,int gridSize,std::string outputFile,
               bool useSSE);
  void Search(int nbThread);
  void FindKeyCPU(TH_PARAM *p);
  void FindKeyGPU();

private:

  std::string GetHex(std::vector<unsigned char> &buffer);
  std::string GetExpectedTime(double keyRate, double keyCount);
  bool checkAddr(std::string &addr, Int &key, uint64_t incr);
  void output(std::string addr, std::string pAddr, std::string pAddrHex, std::string chkAddr, std::string chkAddrC);

  Secp256K1 secp;
  double _difficulty;
  std::string vanityPrefix;
  prefix_t sPrefix;
  Int startKey;
  uint64_t counters[256];
  double startTime;
  bool searchComp;
  bool useGpu;
  int gpuId;
  bool stopWhenFound;
  bool endOfSearch;
  int gridSize;
  int nbCpuThread;
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