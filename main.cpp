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
#include <fstream>
#include <string>
#include <string.h>
#include <stdexcept>

#define RELEASE "1.8"

using namespace std;

// ------------------------------------------------------------------------------------------

void printUsage() {

  printf("VanitySeacrh [-check] [-v] [-u] [-gpu] [-stop] [-i inputfile] [-o outputfile] [-gpuId gpuId1[,gpuId2,...]] [-g gridSize1[,gridSize2,...]] [-s seed] [-t threadNumber] prefix\n");
  printf(" prefix: prefix to search\n");
  printf(" -v: Print version\n");
  printf(" -check: Check CPU and GPU kernel vs CPU\n");
  printf(" -u: Search uncompressed addresses\n");
  printf(" -b: Search both uncompressed or compressed addresses\n");
  printf(" -o outputfile: Output results to the specified file\n");
  printf(" -i inputfile: Get list of prefixes to search from specified file\n");
  printf(" -gpu: Enable gpu calculation\n");
  printf(" -gpu gpuId1,gpuId2,...: List of GPU(s) to use, default is 0\n");
  printf(" -g gridSize1,gridSize2,...: Specify GPU(s) kernel gridsize, default is 8*(MP number)\n");
  printf(" -s seed: Specify a seed for the base key, default is random\n");
  printf(" -t threadNumber: Specify number of CPU thread, default is number of core\n");
  printf(" -nosse : Disable SSE hash function\n");
  printf(" -l : List cuda enabled devices\n");
  printf(" -stop: Stop when all prefixes are found\n");
  exit(-1);

}

// ------------------------------------------------------------------------------------------

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

// ------------------------------------------------------------------------------------------

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

// ------------------------------------------------------------------------------------------

void parseFile(string fileName, vector<string> &lines) {
  
  // Get file size
  FILE *fp = fopen(fileName.c_str(),"rb");
  if (fp == NULL) {
    printf("Error: Cannot open %s %s\n",fileName.c_str(),strerror(errno));
  }
  fseek(fp, 0L, SEEK_END);
  size_t sz = ftell(fp);
  size_t nbAddr = sz / 33; /* Upper approximation */
  bool loaddingProgress = sz>100000;
  fclose(fp);

  int nbLine = 0;
  string line;
  ifstream inFile(fileName);
  lines.reserve(nbAddr);
  while (getline(inFile, line)) {
	  
	// Remove ending \r\n
	int l = line.length()-1;
	while(l>=0 && isspace(line.at(l))) {
	  line.pop_back();
	  l--;
    }
	  
    lines.push_back(line);
    nbLine++;
    if (loaddingProgress) {
      if ((nbLine % 50000)==0)
        printf("[Loading input file %5.1f%%]\r",((double)nbLine*100.0)/((double)(nbAddr)*33.0/34.0));
    }
  }

  if (loaddingProgress)
    printf("[Loading input file 100.0%%]\n");

}

// ------------------------------------------------------------------------------------------

int main(int argc, char* argv[]) {

  // Global Init
  Timer::Init();
  rseed((unsigned long)time(NULL));

  // Init SecpK1
  Secp256K1 secp;
  secp.Init();

  /*
  FILE *f = fopen("addr_25M.txt","w");
  for (int i = 0; i < 25000000; i++) {
    Int k;
    k.Rand(256);
    Point p = secp.ComputePublicKey(&k);
    string addr = secp.GetAddress(p,true);
    fprintf(f,"%s\n",addr.c_str());
    if (i % 100000 == 0) {
      printf("%d\n",i);
    }
  }
  fclose(f);
  exit(0);
  */

  // Browse arguments
  if (argc < 2) {
    printf("Not enough argument\n");
    printUsage();
  }

  int a = 1;
  bool gpuEnable = false;
  bool stop = false;
  int searchMode = SEARCH_COMPRESSED;
  vector<int> gpuId = {0};
  vector<int> gridSize = {-1};
  string seed = "";
  vector<string> prefix;
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
      g.SetSearchMode(searchMode);
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
      searchMode = SEARCH_UNCOMPRESSED;
      a++;
    } else if (strcmp(argv[a], "-b") == 0) {
      searchMode = SEARCH_BOTH;
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
    } else if (strcmp(argv[a], "-i") == 0) {
      a++;
      parseFile(string(argv[a]),prefix);
      a++;
    } else if (strcmp(argv[a], "-t") == 0) {
      a++;
      nbCPUThread = getInt("nbCPUThread",argv[a]);
      a++;
      tSpecified = true;
    } else if (a == argc - 1) {
      prefix.push_back(string(argv[a]));
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

  VanitySearch *v = new VanitySearch(secp, prefix, seed,searchMode,gpuEnable,stop,outputFile,sse);
  v->Search(nbCPUThread,gpuId,gridSize);

  return 0;
}
