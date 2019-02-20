#ifndef GPUENGINEH
#define GPUENGINEH

#include <vector>
#include <cuda_runtime.h>
#include "../SECP256K1.h"

// Number of key per thread (must be a multiple of GRP_SIZE) per kernel call
#define STEP_SIZE 256

// Number of thread per block
#define NB_TRHEAD_PER_GROUP 64

// Maximum number of prefix found per thread (MAX_FOUND<=STEP_SIZE)
// If MAX_FOUND is too large it may crash and slow down the kernel
// Probability to lost 1 (or more) can be calculated using Psk
#define MAX_FOUND 4 // Plost(256,4) = 2.3e-09 (very unlikely)

#define ITEM_SIZE 22
#define MEMOUT_PER_THREAD (1+MAX_FOUND*ITEM_SIZE)

typedef uint16_t prefix_t;

typedef struct {
  uint32_t thId;
  uint16_t incr;
  uint8_t  *hash;
} ITEM;

class GPUEngine {

public:

  GPUEngine(int nbThreadGroup,int gpuId); 
  ~GPUEngine();
  void SetPrefix(prefix_t prefix);
  bool SetKeys(Point *p);
  void SetSearchMode(bool compressed);
  bool Launch(std::vector<ITEM> &prefixFound,bool spinWait=false);
  int GetNbThread();

  bool Check(Secp256K1 &secp);
  std::string deviceName;
  static void GenerateCode(Secp256K1 &secp, int size);

private:

  bool callKernel();
  static void ComputeIndex(std::vector<int> &s, int depth, int n);
  static void Browse(FILE *f,int depth, int max, int s);

  int nbThread;
  prefix_t prefix;
  uint64_t *inputKey;
  uint64_t *inputKeyPinned;
  uint8_t *outputPrefix;
  uint8_t *outputPrefixPinned;
  bool initialised;
  bool searchComp;

};

#endif // GPUENGINEH
