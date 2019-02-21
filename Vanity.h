#ifndef VANITYH
#define VANITYH

#include <windows.h>
#include <string>
#include <vector>
#include "SECP256K1.h"
#include "GPU/GPUEngine.h"

class VanitySearch;

typedef struct {

  VanitySearch *obj;
  int threadId;

} TH_PARAM;

class VanitySearch {

public:

  VanitySearch(Secp256K1 &secp, std::string prefix, std::string seed, bool compressed, bool useGpu, int gpuId, bool stop,int gridSize);
  void Search(int nbThread);
  void FindKeyCPU(TH_PARAM *p);
  void FindKeyGPU();

private:

  std::string GetHex(std::vector<unsigned char> &buffer);
  std::string GetExpectedTime(double keyRate, double keyCount);
  bool checkAddr(std::string &addr, Int &key, uint64_t incr);

  Secp256K1 secp;
  double _difficulty;
  std::string vanityPrefix;
  prefix_t sPrefix;
  Int startKey;
  HANDLE ghMutex;
  uint64_t counters[256];
  double startTime;
  bool searchComp;
  bool useGpu;
  int gpuId;
  bool stopWhenFound;
  bool endOfSearch;
  int gridSize;
  int nbCpuThread;

};

#endif // VANITYH