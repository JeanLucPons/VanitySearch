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

#include "Timer.h"
#include "Vanity.h"
#include "SECP256k1.h"
#include <string>
#include <string.h>
#include <stdexcept>

#define RELEASE "1.5"

using namespace std;

void printUsage() {

  printf("VanitySeacrh [-check] [-v] [-u] [-gpu] [-stop] [-o outputfile] [-gpuId gpuId] [-g gridSize] [-s seed] [-t threadNumber] prefix\n");
  printf(" prefix: prefix to search\n");
  printf(" -v: Print version\n");
  printf(" -check: Check CPU and GPU kernel vs CPU\n");
  printf(" -u: Search uncompressed addresses\n");
  printf(" -o outputfile: Output results to the specified file\n");
  printf(" -gpu: Enable gpu calculation\n");
  printf(" -gpu gpuId1,gpuId2,...: List of GPU(s) to use, default is 0\n");
  printf(" -g gridSize1,gridSize2,...: Specify GPU(s) kernel gridsize, default is 16*(MP number)\n");
  printf(" -s seed: Specify a seed for the base key, default is random\n");
  printf(" -t threadNumber: Specify number of CPU thread, default is number of core\n");
  printf(" -nosse : Disable SSE hash function\n");
  printf(" -l : List cuda enabled devices\n");
  printf(" -stop: Stop when prefix is found\n");
  exit(-1);

}

int getInt(string name,char *v) {

  int r;

  try {

    r = std::stoi(string(v));

  } catch(std::invalid_argument&) {

    printf("Invalid %s argument, number expected\n",name.c_str());
    exit(-1);

  }

  return r;

}

void getInts(string name,vector<int> &tokens, const string &text, char sep) {

  size_t start = 0, end = 0;
  tokens.clear();
  int item;

  try {

    while ((end = text.find(sep, start)) != string::npos) {
      item = std::stoi(text.substr(start, end - start));
      tokens.push_back(item);
      start = end + 1;
    }

    item = std::stoi(text.substr(start));
    tokens.push_back(item);

  } catch(std::invalid_argument &) {

    printf("Invalid %s argument, number expected\n",name.c_str());
    exit(-1);

  }

}


int main(int argc, char* argv[]) {

  // Global Init
  Timer::Init();
  rseed((unsigned long)time(NULL));

  // Init SecpK1
  Secp256K1 secp;
  secp.Init();

  // Browse arguments
  if (argc < 2) {
    printf("Not enough argument\n");
    printUsage();
  }

  int a = 1;
  bool gpuEnable = false;
  bool stop = false;
  bool uncomp = false;
  vector<int> gpuId = {0};
  vector<int> gridSize = {-1};
  string seed = "";
  string prefix = "";
  string outputFile = "";
  int nbCPUThread = Timer::getCoreNumber();
  bool tSpecified = false;
  bool sse = true;

  while (a < argc) {

    if (strcmp(argv[a], "-gpu")==0) {
      gpuEnable = true;
      a++;
    } else if (strcmp(argv[a], "-gpuId")==0) {
      a++;
      getInts("gpuId",gpuId,string(argv[a]),',');
      a++;
    } else if (strcmp(argv[a], "-stop") == 0) {
      stop = true;
      a++;
    } else if (strcmp(argv[a], "-v") == 0) {
      printf("%s\n",RELEASE);
      exit(0);
    } else if (strcmp(argv[a], "-check") == 0) {

      Int::Check();
      secp.Check();

#ifdef WITHGPU
      GPUEngine g(gridSize[0],gpuId[0]);
      g.SetSearchMode(!uncomp);
      g.Check(secp);
#else
  printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif
      exit(0);
    } else if (strcmp(argv[a], "-l") == 0) {

#ifdef WITHGPU
      GPUEngine::PrintCudaInfo();
#else
  printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif
      exit(0);

    } else if (strcmp(argv[a], "-u") == 0) {
      uncomp = true;
      a++;
    } else if (strcmp(argv[a], "-nosse") == 0) {
      sse = false;
      a++;
    } else if (strcmp(argv[a], "-g") == 0) {
      a++;
      getInts("gridSize",gridSize,string(argv[a]),',');
      a++;
    } else if (strcmp(argv[a], "-s") == 0) {
      a++;
      seed = string(argv[a]);
      a++;
    } else if (strcmp(argv[a], "-o") == 0) {
      a++;
      outputFile = string(argv[a]);
      a++;
    } else if (strcmp(argv[a], "-t") == 0) {
      a++;
      nbCPUThread = getInt("nbCPUThread",argv[a]);
      a++;
      tSpecified = true;
    } else if (a == argc - 1) {
      prefix = string(argv[a]);
      a++;
    } else {
      printf("Unexpected %s argument\n",argv[a]);
      printUsage();
    }

  }

  if(gpuId.size()!=gridSize.size()) {
    if(gridSize.size()==1 && gridSize[0]==-1) {
      gridSize.clear();
      for(int i=0;i<gpuId.size();i++)
        gridSize.push_back(-1);
    } else {
      printf("Invalid gridSize or gpuId argument, must have same size\n");
      printUsage();
    }
  }

  // Let one CPU core free if gpu is enabled
  // It will avoid to hang the system
  if( !tSpecified && nbCPUThread>1 && gpuEnable)
    nbCPUThread--;

  VanitySearch *v = new VanitySearch(secp, prefix, seed,!uncomp,gpuEnable,stop,outputFile,sse);
  v->Search(nbCPUThread,gpuId,gridSize);

  return 0;
}
