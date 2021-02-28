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

#ifndef WIN64
#include <unistd.h>
#include <stdio.h>
#endif

#include "GPUEngine.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdint.h>
#include "../hash/sha256.h"
#include "../hash/ripemd160.h"
#include "../Timer.h"

#include "GPUGroup.h"
#include "GPUMath.h"
#include "GPUHash.h"
#include "GPUBase58.h"
#include "GPUWildcard.h"
#include "GPUCompute.h"

// ---------------------------------------------------------------------------------------

__global__ void comp_keys(uint32_t mode,prefix_t *prefix, uint32_t *lookup32, uint64_t *keys, uint32_t maxFound, uint32_t *found) {

  int xPtr = (blockIdx.x*blockDim.x) * 8;
  int yPtr = xPtr + 4 * blockDim.x;
  ComputeKeys(mode, keys + xPtr, keys + yPtr, prefix, lookup32, maxFound, found);

}

__global__ void comp_keys_p2sh(uint32_t mode, prefix_t *prefix, uint32_t *lookup32, uint64_t *keys, uint32_t maxFound, uint32_t *found) {

  int xPtr = (blockIdx.x*blockDim.x) * 8;
  int yPtr = xPtr + 4 * blockDim.x;
  ComputeKeysP2SH(mode, keys + xPtr, keys + yPtr, prefix, lookup32, maxFound, found);

}

__global__ void comp_keys_comp(prefix_t *prefix, uint32_t *lookup32, uint64_t *keys, uint32_t maxFound, uint32_t *found) {

  int xPtr = (blockIdx.x*blockDim.x) * 8;
  int yPtr = xPtr + 4 * blockDim.x;
  ComputeKeysComp(keys + xPtr, keys + yPtr, prefix, lookup32, maxFound, found);

}

__global__ void comp_keys_pattern(uint32_t mode, prefix_t *pattern, uint64_t *keys,  uint32_t maxFound, uint32_t *found) {

  int xPtr = (blockIdx.x*blockDim.x) * 8;
  int yPtr = xPtr + 4 * blockDim.x;
  ComputeKeys(mode, keys + xPtr, keys + yPtr, NULL, (uint32_t *)pattern, maxFound, found);

}

__global__ void comp_keys_p2sh_pattern(uint32_t mode, prefix_t *pattern, uint64_t *keys, uint32_t maxFound, uint32_t *found) {

  int xPtr = (blockIdx.x*blockDim.x) * 8;
  int yPtr = xPtr + 4 * blockDim.x;
  ComputeKeysP2SH(mode, keys + xPtr, keys + yPtr, NULL, (uint32_t *)pattern, maxFound, found);

}

//#define FULLCHECK
#ifdef FULLCHECK

// ---------------------------------------------------------------------------------------

__global__ void chekc_mult(uint64_t *a, uint64_t *b, uint64_t *r) {

  _ModMult(r, a, b);
  r[4]=0;

}

// ---------------------------------------------------------------------------------------

__global__ void chekc_hash160(uint64_t *x, uint64_t *y, uint32_t *h) {

  _GetHash160(x, y, (uint8_t *)h);
  _GetHash160Comp(x, y, (uint8_t *)(h+5));

}

// ---------------------------------------------------------------------------------------

__global__ void get_endianness(uint32_t *endian) {

  uint32_t a = 0x01020304;
  uint8_t fb = *(uint8_t *)(&a);
  *endian = (fb==0x04);

}

#endif //FULLCHECK

// ---------------------------------------------------------------------------------------

using namespace std;

std::string toHex(unsigned char *data, int length) {

  string ret;
  char tmp[3];
  for (int i = 0; i < length; i++) {
    if (i && i % 4 == 0) ret.append(" ");
    sprintf(tmp, "%02x", (int)data[i]);
    ret.append(tmp);
  }
  return ret;

}

int _ConvertSMVer2Cores(int major, int minor) {

  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x20, 32}, // Fermi Generation (SM 2.0) GF100 class
      {0x21, 48}, // Fermi Generation (SM 2.1) GF10x class
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {-1, -1} };

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  return 0;

}

GPUEngine::GPUEngine(int nbThreadGroup, int nbThreadPerGroup, int gpuId, uint32_t maxFound,bool rekey) {

  // Initialise CUDA
  this->rekey = rekey;
  this->nbThreadPerGroup = nbThreadPerGroup;
  initialised = false;
  cudaError_t err;

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("GPUEngine: CudaGetDeviceCount %s %d\n", cudaGetErrorString(error_id),error_id);
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  err = cudaSetDevice(gpuId);
  if (err != cudaSuccess) {
    printf("GPUEngine: %s\n", cudaGetErrorString(err));
    return;
  }

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, gpuId);

  if (nbThreadGroup == -1)
    nbThreadGroup = deviceProp.multiProcessorCount * 8;

  this->nbThread = nbThreadGroup * nbThreadPerGroup;
  this->maxFound = maxFound;
  this->outputSize = (maxFound*ITEM_SIZE + 4);

  char tmp[512];
  sprintf(tmp,"GPU #%d %s (%dx%d cores) Grid(%dx%d)",
  gpuId,deviceProp.name,deviceProp.multiProcessorCount,
  _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
                      nbThread / nbThreadPerGroup,
                      nbThreadPerGroup);
  deviceName = std::string(tmp);

  // Prefer L1 (We do not use __shared__ at all)
  err = cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  if (err != cudaSuccess) {
    printf("GPUEngine: %s\n", cudaGetErrorString(err));
    return;
  }

  size_t stackSize = 49152;
  err = cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
  if (err != cudaSuccess) {
    printf("GPUEngine: %s\n", cudaGetErrorString(err));
    return;
  }

  /*
  size_t heapSize = ;
  err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
  if (err != cudaSuccess) {
    printf("Error: %s\n", cudaGetErrorString(err));
    exit(0);
  }

  size_t size;
  cudaDeviceGetLimit(&size, cudaLimitStackSize);
  printf("Stack Size %lld\n", size);
  cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
  printf("Heap Size %lld\n", size);
  */

  // Allocate memory
  err = cudaMalloc((void **)&inputPrefix, _64K * 2);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate prefix memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&inputPrefixPinned, _64K * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate prefix pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaMalloc((void **)&inputKey, nbThread * 32 * 2);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate input memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&inputKeyPinned, nbThread * 32 * 2, cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate input pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaMalloc((void **)&outputPrefix, outputSize);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate output memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&outputPrefixPinned, outputSize, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate output pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }

  searchMode = SEARCH_COMPRESSED;
  searchType = P2PKH;
  initialised = true;
  pattern = "";
  hasPattern = false;
  inputPrefixLookUp = NULL;

}

int GPUEngine::GetGroupSize() {
  return GRP_SIZE;
}

void GPUEngine::PrintCudaInfo() {

  cudaError_t err;

  const char *sComputeMode[] =
  {
    "Multiple host threads",
    "Only one host thread",
    "No host thread",
    "Multiple process threads",
    "Unknown",
     NULL
  };

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    printf("GPUEngine: CudaGetDeviceCount %s\n", cudaGetErrorString(error_id));
    return;
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    printf("GPUEngine: There are no available device(s) that support CUDA\n");
    return;
  }

  for(int i=0;i<deviceCount;i++) {

    err = cudaSetDevice(i);
    if (err != cudaSuccess) {
      printf("GPUEngine: cudaSetDevice(%d) %s\n", i, cudaGetErrorString(err));
      return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, i);
    printf("GPU #%d %s (%dx%d cores) (Cap %d.%d) (%.1f MB) (%s)\n",
      i,deviceProp.name,deviceProp.multiProcessorCount,
      _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
      deviceProp.major, deviceProp.minor,(double)deviceProp.totalGlobalMem/1048576.0,
      sComputeMode[deviceProp.computeMode]);

  }

}

GPUEngine::~GPUEngine() {

  cudaFree(inputKey);
  cudaFree(inputPrefix);
  if(inputPrefixLookUp) cudaFree(inputPrefixLookUp);
  cudaFreeHost(outputPrefixPinned);
  cudaFree(outputPrefix);

}

int GPUEngine::GetNbThread() {
  return nbThread;
}

void GPUEngine::SetSearchMode(int searchMode) {
  this->searchMode = searchMode;
}

void GPUEngine::SetSearchType(int searchType) {
  this->searchType = searchType;
}

void GPUEngine::SetPrefix(std::vector<prefix_t> prefixes) {

  memset(inputPrefixPinned, 0, _64K * 2);
  for(int i=0;i<(int)prefixes.size();i++)
    inputPrefixPinned[prefixes[i]]=1;

  // Fill device memory
  cudaMemcpy(inputPrefix, inputPrefixPinned, _64K * 2, cudaMemcpyHostToDevice);

  // We do not need the input pinned memory anymore
  cudaFreeHost(inputPrefixPinned);
  inputPrefixPinned = NULL;
  lostWarning = false;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: SetPrefix: %s\n", cudaGetErrorString(err));
  }

}

void GPUEngine::SetPattern(const char *pattern) {

  strcpy((char *)inputPrefixPinned,pattern);

  // Fill device memory
  cudaMemcpy(inputPrefix, inputPrefixPinned, _64K * 2, cudaMemcpyHostToDevice);

  // We do not need the input pinned memory anymore
  cudaFreeHost(inputPrefixPinned);
  inputPrefixPinned = NULL;
  lostWarning = false;

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: SetPattern: %s\n", cudaGetErrorString(err));
  }

  hasPattern = true;

}

void GPUEngine::SetPrefix(std::vector<LPREFIX> prefixes, uint32_t totalPrefix) {

  // Allocate memory for the second level of lookup tables
  cudaError_t err = cudaMalloc((void **)&inputPrefixLookUp, (_64K+totalPrefix) * 4);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate prefix lookup memory: %s\n", cudaGetErrorString(err));
    return;
  }
  err = cudaHostAlloc(&inputPrefixLookUpPinned, (_64K+totalPrefix) * 4, cudaHostAllocWriteCombined | cudaHostAllocMapped);
  if (err != cudaSuccess) {
    printf("GPUEngine: Allocate prefix lookup pinned memory: %s\n", cudaGetErrorString(err));
    return;
  }

  uint32_t offset = _64K;
  memset(inputPrefixPinned, 0, _64K * 2);
  memset(inputPrefixLookUpPinned, 0, _64K * 4);
  for (int i = 0; i < (int)prefixes.size(); i++) {
    int nbLPrefix = (int)prefixes[i].lPrefixes.size();
    inputPrefixPinned[prefixes[i].sPrefix] = (uint16_t)nbLPrefix;
    inputPrefixLookUpPinned[prefixes[i].sPrefix] = offset;
    for (int j = 0; j < nbLPrefix; j++) {
      inputPrefixLookUpPinned[offset++]=prefixes[i].lPrefixes[j];
    }
  }

  if (offset != (_64K+totalPrefix)) {
    printf("GPUEngine: Wrong totalPrefix %d!=%d!\n",offset- _64K, totalPrefix);
    return;
  }

  // Fill device memory
  cudaMemcpy(inputPrefix, inputPrefixPinned, _64K * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(inputPrefixLookUp, inputPrefixLookUpPinned, (_64K+totalPrefix) * 4, cudaMemcpyHostToDevice);

  // We do not need the input pinned memory anymore
  cudaFreeHost(inputPrefixPinned);
  inputPrefixPinned = NULL;
  cudaFreeHost(inputPrefixLookUpPinned);
  inputPrefixLookUpPinned = NULL;
  lostWarning = false;

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: SetPrefix (large): %s\n", cudaGetErrorString(err));
  }

}

bool GPUEngine::callKernel() {

  // Reset nbFound
  cudaMemset(outputPrefix,0,4);

  // Call the kernel (Perform STEP_SIZE keys per thread)
  if (searchType == P2SH) {

    if (hasPattern) {
      comp_keys_p2sh_pattern << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
        (searchMode, inputPrefix, inputKey, maxFound, outputPrefix);
    } else {
      comp_keys_p2sh << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
        (searchMode, inputPrefix, inputPrefixLookUp, inputKey, maxFound, outputPrefix);
    }

  } else {

    // P2PKH or BECH32
    if (hasPattern) {
      if (searchType == BECH32) {
        // TODO
        printf("GPUEngine: (TODO) BECH32 not yet supported with wildard\n");
        return false;
      }
      comp_keys_pattern << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
        (searchMode, inputPrefix, inputKey, maxFound, outputPrefix);
    } else {
      if (searchMode == SEARCH_COMPRESSED) {
        comp_keys_comp << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
          (inputPrefix, inputPrefixLookUp, inputKey, maxFound, outputPrefix);
      } else {
        comp_keys << < nbThread / nbThreadPerGroup, nbThreadPerGroup >> >
          (searchMode, inputPrefix, inputPrefixLookUp, inputKey, maxFound, outputPrefix);
      }
    }

  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: Kernel: %s\n", cudaGetErrorString(err));
    return false;
  }
  return true;

}

bool GPUEngine::SetKeys(Point *p) {

  // Sets the starting keys for each thread
  // p must contains nbThread public keys
  for (int i = 0; i < nbThread; i+= nbThreadPerGroup) {
    for (int j = 0; j < nbThreadPerGroup; j++) {

      inputKeyPinned[8*i + j + 0* nbThreadPerGroup] = p[i + j].x.bits64[0];
      inputKeyPinned[8*i + j + 1* nbThreadPerGroup] = p[i + j].x.bits64[1];
      inputKeyPinned[8*i + j + 2* nbThreadPerGroup] = p[i + j].x.bits64[2];
      inputKeyPinned[8*i + j + 3* nbThreadPerGroup] = p[i + j].x.bits64[3];

      inputKeyPinned[8*i + j + 4* nbThreadPerGroup] = p[i + j].y.bits64[0];
      inputKeyPinned[8*i + j + 5* nbThreadPerGroup] = p[i + j].y.bits64[1];
      inputKeyPinned[8*i + j + 6* nbThreadPerGroup] = p[i + j].y.bits64[2];
      inputKeyPinned[8*i + j + 7* nbThreadPerGroup] = p[i + j].y.bits64[3];

    }
  }

  // Fill device memory
  cudaMemcpy(inputKey, inputKeyPinned, nbThread*32*2, cudaMemcpyHostToDevice);

  if (!rekey) {
    // We do not need the input pinned memory anymore
    cudaFreeHost(inputKeyPinned);
    inputKeyPinned = NULL;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: SetKeys: %s\n", cudaGetErrorString(err));
  }

  return callKernel();

}

bool GPUEngine::Launch(std::vector<ITEM> &prefixFound,bool spinWait) {


  prefixFound.clear();

  // Get the result

  if(spinWait) {

    cudaMemcpy(outputPrefixPinned, outputPrefix, outputSize, cudaMemcpyDeviceToHost);

  } else {

    // Use cudaMemcpyAsync to avoid default spin wait of cudaMemcpy wich takes 100% CPU
    cudaEvent_t evt;
    cudaEventCreate(&evt);
    cudaMemcpyAsync(outputPrefixPinned, outputPrefix, 4, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(evt, 0);
    while (cudaEventQuery(evt) == cudaErrorNotReady) {
      // Sleep 1 ms to free the CPU
      Timer::SleepMillis(1);
    }
    cudaEventDestroy(evt);

  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: Launch: %s\n", cudaGetErrorString(err));
    return false;
  }

  // Look for prefix found
  uint32_t nbFound = outputPrefixPinned[0];
  if (nbFound > maxFound) {
    // prefix has been lost
    if (!lostWarning) {
      printf("\nWarning, %d items lost\nHint: Search with less prefixes, less threads (-g) or increase maxFound (-m)\n", (nbFound - maxFound));
      lostWarning = true;
    }
    nbFound = maxFound;
  }

  // When can perform a standard copy, the kernel is eneded
  cudaMemcpy( outputPrefixPinned , outputPrefix , nbFound*ITEM_SIZE + 4 , cudaMemcpyDeviceToHost);

  for (uint32_t i = 0; i < nbFound; i++) {
    uint32_t *itemPtr = outputPrefixPinned + (i*ITEM_SIZE32 + 1);
    ITEM it;
    it.thId = itemPtr[0];
    int16_t *ptr = (int16_t *)&(itemPtr[1]);
    it.endo = ptr[0] & 0x7FFF;
    it.mode = (ptr[0]&0x8000)!=0;
    it.incr = ptr[1];
    it.hash = (uint8_t *)(itemPtr + 2);
    prefixFound.push_back(it);
  }

  return callKernel();

}

bool GPUEngine::CheckHash(uint8_t *h, vector<ITEM>& found,int tid,int incr,int endo, int *nbOK) {

  bool ok = true;

  // Search in found by GPU
  bool f = false;
  int l = 0;
  //printf("Search: %s\n", toHex(h,20).c_str());
  while (l < found.size() && !f) {
    f = ripemd160_comp_hash(found[l].hash, h);
    if (!f) l++;
  }
  if (f) {
    found.erase(found.begin() + l);
    *nbOK = *nbOK+1;
  } else {
    ok = false;
    printf("Expected item not found %s (thread=%d, incr=%d, endo=%d)\n",
      toHex(h, 20).c_str(),tid,incr,endo);
  }

  return ok;

}

bool GPUEngine::Check(Secp256K1 *secp) {

  uint8_t h[20];
  int i = 0;
  int j = 0;
  bool ok = true;

  if(!initialised)
    return false;

  printf("GPU: %s\n",deviceName.c_str());

#ifdef FULLCHECK

  // Get endianess
  get_endianness<<<1,1>>>(outputPrefix);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("GPUEngine: get_endianness: %s\n", cudaGetErrorString(err));
    return false;
  }
  cudaMemcpy(outputPrefixPinned, outputPrefix,1,cudaMemcpyDeviceToHost);
  littleEndian = *outputPrefixPinned != 0;
  printf("Endianness: %s\n",(littleEndian?"Little":"Big"));

  // Check modular mult
  Int a;
  Int b;
  Int r;
  Int c;
  a.Rand(256);
  b.Rand(256);
  c.ModMulK1(&a,&b);
  memcpy(inputKeyPinned,a.bits64,BIFULLSIZE);
  memcpy(inputKeyPinned+5,b.bits64,BIFULLSIZE);
  cudaMemcpy(inputKey, inputKeyPinned, BIFULLSIZE*2, cudaMemcpyHostToDevice);
  chekc_mult<<<1,1>>>(inputKey,inputKey+5,(uint64_t *)outputPrefix);
  cudaMemcpy(outputPrefixPinned, outputPrefix, BIFULLSIZE, cudaMemcpyDeviceToHost);
  memcpy(r.bits64,outputPrefixPinned,BIFULLSIZE);

  if(!c.IsEqual(&r)) {
    printf("\nModular Mult wrong:\nR=%s\nC=%s\n",
    toHex((uint8_t *)r.bits64,BIFULLSIZE).c_str(),
    toHex((uint8_t *)c.bits64,BIFULLSIZE).c_str());
    return false;
  }

  // Check hash 160C
  uint8_t hc[20];
  Point pi;
  pi.x.Rand(256);
  pi.y.Rand(256);
  secp.GetHash160(pi, false, h);
  secp.GetHash160(pi, true, hc);
  memcpy(inputKeyPinned,pi.x.bits64,BIFULLSIZE);
  memcpy(inputKeyPinned+5,pi.y.bits64,BIFULLSIZE);
  cudaMemcpy(inputKey, inputKeyPinned, BIFULLSIZE*2, cudaMemcpyHostToDevice);
  chekc_hash160<<<1,1>>>(inputKey,inputKey+5,outputPrefix);
  cudaMemcpy(outputPrefixPinned, outputPrefix, 64, cudaMemcpyDeviceToHost);

  if(!ripemd160_comp_hash((uint8_t *)outputPrefixPinned,h)) {
    printf("\nGetHask160 wrong:\n%s\n%s\n",
    toHex((uint8_t *)outputPrefixPinned,20).c_str(),
    toHex(h,20).c_str());
    return false;
  }
  if (!ripemd160_comp_hash((uint8_t *)(outputPrefixPinned+5), hc)) {
    printf("\nGetHask160Comp wrong:\n%s\n%s\n",
      toHex((uint8_t *)(outputPrefixPinned + 5), 20).c_str(),
      toHex(h, 20).c_str());
    return false;
  }

#endif //FULLCHECK

  Point *p = new Point[nbThread];
  Point *p2 = new Point[nbThread];
  Int k;

  // Check kernel
  int nbFoundCPU[6];
  int nbOK[6];
  vector<ITEM> found;
  bool searchComp;

  if (searchMode == SEARCH_BOTH) {
    printf("Warning, Check function does not support BOTH_MODE, use either compressed or uncompressed");
    return true;
  }

  searchComp = (searchMode == SEARCH_COMPRESSED)?true:false;

  uint32_t seed = (uint32_t)time(NULL);
  printf("Seed: %u\n",seed);
  rseed(seed);
  memset(nbOK,0,sizeof(nbOK));
  memset(nbFoundCPU, 0, sizeof(nbFoundCPU));
  for (int i = 0; i < nbThread; i++) {
    k.Rand(256);
    p[i] = secp->ComputePublicKey(&k);
    // Group starts at the middle
    k.Add((uint64_t)GRP_SIZE/2);
    p2[i] = secp->ComputePublicKey(&k);
  }

  std::vector<prefix_t> prefs;
  prefs.push_back(0xFEFE);
  prefs.push_back(0x1234);
  SetPrefix(prefs);
  SetKeys(p2);
  double t0 = Timer::get_tick();
  Launch(found,true);
  double t1 = Timer::get_tick();
  Timer::printResult((char *)"Key", 6*STEP_SIZE*nbThread, t0, t1);

  //for (int i = 0; i < found.size(); i++) {
  //  printf("[%d]: thId=%d incr=%d\n", i, found[i].thId,found[i].incr);
  //  printf("[%d]: %s\n", i,toHex(found[i].hash,20).c_str());
  //}

  printf("ComputeKeys() found %d items , CPU check...\n",(int)found.size());

  Int beta,beta2;
  beta.SetBase16((char *)"7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee");
  beta2.SetBase16((char *)"851695d49a83f8ef919bb86153cbcb16630fb68aed0a766a3ec693d68e6afa40");

  // Check with CPU
  for (j = 0; (j<nbThread); j++) {
    for (i = 0; i < STEP_SIZE; i++) {

      Point pt,p1,p2;
      pt = p[j];
      p1 = p[j];
      p2 = p[j];
      p1.x.ModMulK1(&beta);
      p2.x.ModMulK1(&beta2);
      p[j] = secp->NextKey(p[j]);

      // Point and endo
      secp->GetHash160(P2PKH, searchComp, pt, h);
      prefix_t pr = *(prefix_t *)h;
      if (pr == 0xFEFE || pr == 0x1234) {
	      nbFoundCPU[0]++;
        ok &= CheckHash(h,found, j, i, 0, nbOK + 0);
      }
      secp->GetHash160(P2PKH, searchComp, p1, h);
      pr = *(prefix_t *)h;
      if (pr == 0xFEFE || pr == 0x1234) {
        nbFoundCPU[1]++;
        ok &= CheckHash(h, found, j, i, 1, nbOK + 1);
      }
      secp->GetHash160(P2PKH, searchComp, p2, h);
      pr = *(prefix_t *)h;
      if (pr == 0xFEFE || pr == 0x1234) {
        nbFoundCPU[2]++;
        ok &= CheckHash(h, found, j, i, 2, nbOK + 2);
      }

      // Symetrics
      pt.y.ModNeg();
      p1.y.ModNeg();
      p2.y.ModNeg();

      secp->GetHash160(P2PKH, searchComp, pt, h);
      pr = *(prefix_t *)h;
      if (pr == 0xFEFE || pr == 0x1234) {
        nbFoundCPU[3]++;
        ok &= CheckHash(h, found, j, -i, 0, nbOK + 3);
      }
      secp->GetHash160(P2PKH, searchComp, p1, h);
      pr = *(prefix_t *)h;
      if (pr == 0xFEFE || pr == 0x1234) {
        nbFoundCPU[4]++;
        ok &= CheckHash(h, found, j, -i, 1, nbOK + 4);
      }
      secp->GetHash160(P2PKH, searchComp, p2, h);
      pr = *(prefix_t *)h;
      if (pr == 0xFEFE || pr == 0x1234) {
        nbFoundCPU[5]++;
        ok &= CheckHash(h, found, j, -i, 2, nbOK + 5);
      }

    }
  }

  if (ok && found.size()!=0) {
    ok = false;
    printf("Unexpected item found !\n");
  }

  if( !ok ) {

    int nbF = nbFoundCPU[0] + nbFoundCPU[1] + nbFoundCPU[2] +
              nbFoundCPU[3] + nbFoundCPU[4] + nbFoundCPU[5];
    printf("CPU found %d items\n",nbF);

    printf("GPU: point   correct [%d/%d]\n", nbOK[0] , nbFoundCPU[0]);
    printf("GPU: endo #1 correct [%d/%d]\n", nbOK[1] , nbFoundCPU[1]);
    printf("GPU: endo #2 correct [%d/%d]\n", nbOK[2] , nbFoundCPU[2]);

    printf("GPU: sym/point   correct [%d/%d]\n", nbOK[3] , nbFoundCPU[3]);
    printf("GPU: sym/endo #1 correct [%d/%d]\n", nbOK[4] , nbFoundCPU[4]);
    printf("GPU: sym/endo #2 correct [%d/%d]\n", nbOK[5] , nbFoundCPU[5]);

    printf("GPU/CPU check Failed !\n");

  }

  if(ok) printf("GPU/CPU check OK\n");

  delete[] p;
  return ok;

}


