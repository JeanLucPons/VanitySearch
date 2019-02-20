#include "SECP256K1.h"
#include "Timer.h"
#include "Vanity.h"
#include "Secp256K1.h"

#define RELEASE "1.0"

using namespace std;

void printUsage() {

  printf("VanitySeacrh [-v] [-u] [-gpu] [-stop] [-gpuId gpuId] [-g gridSize] [-s seed] [-t threadNumber] prefix");
  printf(" prefix: prefix to search\n");
  printf(" -v: Print version\n");
  printf(" -u: Search uncompressed address\n");
  printf(" -gpu: Enable gpu calculation\n");
  printf(" -gpu gpuId: Use gpu gpuId, default is 0\n");
  printf(" -g gridSize: Specify GPU kernel gridsize, default is 16*(MP number)\n");
  printf(" -s seed: Specify a seed for the base key, default is random\n");
  printf(" -t threadNumber: Specify number of CPU thread, default is number of core\n");
  printf(" -stop: Stop when prefix is found\n");
  exit(-1);

}

int getInt(char *v) {

  int r = strtol(v, NULL, 10);
  if (errno == EINVAL)
    printUsage();
  return r;

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
    printUsage();
  }

  int a = 1;
  bool gpuEnable = false;
  bool stop = false;
  bool uncomp = false;
  int gpuId = 0;
  int gridSize = -1;
  string seed = "";
  string prefix = "";
  int nbThread = Timer::getCoreNumber();
  

  while (a < argc) {

    if (strcmp(argv[a], "-gpu")==0) {
      gpuEnable = true;
      a++;
    } else if (strcmp(argv[a], "-gpuId")==0) {
      a++;
      gpuId = getInt(argv[a]);
      a++;
    } else if (strcmp(argv[a], "-stop") == 0) {
      stop = true;
      a++;
    } else if (strcmp(argv[a], "-v") == 0) {
      printf("%s\n",RELEASE);
      exit(0);
    } else if (strcmp(argv[a], "-u") == 0) {
      uncomp = true;
      a++;
    } else if (strcmp(argv[a], "-g") == 0) {
      a++;
      gridSize = getInt(argv[a]);
      a++;
    } else if (strcmp(argv[a], "-s") == 0) {
      a++;
      seed = string(argv[a]);
      a++;
    } else if (strcmp(argv[a], "-t") == 0) {
      a++;
      nbThread = getInt(argv[a]);
      a++;
    } else if (a == argc - 1) {
      prefix = string(argv[a]);
      a++;
    } else {
      printUsage();
    }

  }

  VanitySearch v(secp, prefix, seed,!uncomp,gpuEnable,gpuId,stop,gridSize);
  v.Search(nbThread);

  return 0;
}
