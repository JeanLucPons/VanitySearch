#include "Vanity.h"
#include "Base58.h"
#include "hash/sha256.h"
#include "hash/sha512.h"
#include "IntGroup.h"
#include "Timer.h"
#include <string.h>
#include <math.h>
#ifndef WIN64
#include <pthread.h>
#include <unistd.h>
#endif

using namespace std;

Point Gn[CPU_GRP_SIZE];

// ----------------------------------------------------------------------------

VanitySearch::VanitySearch(Secp256K1 &secp,string prefix,string seed,bool comp, bool useGpu, 
                           int gpuId, bool stop, int gridSize, string outputFile) {

  this->vanityPrefix = prefix;
  this->secp = secp;
  this->searchComp = comp;
  this->useGpu = useGpu;
  this->gpuId = gpuId;
  this->stopWhenFound = stop;
  this->gridSize = gridSize;
  this->outputFile = outputFile;
  sPrefix = -1;
  std::vector<unsigned char> result;

  if (prefix.length() < 2) {
    printf("VanitySearch: Invalid prefix !");
    exit(-1);
  }

  if (prefix.data()[0] != '1' ) {
    printf("VanitySearch: Only prefix starting with 1 allowed !");
    exit(-1);
  }

  string dummy1 = prefix;
  int nbDigit = 0;
  bool wrong = false;

  char *ctimeBuff;
  time_t now = time(NULL);
  ctimeBuff = ctime(&now);
  printf("Start %s", ctimeBuff);

  printf("Search: %s\n",dummy1.c_str());

  // Search for highest hash160 16bit prefix (most probable)

  while (result.size() < 25 && !wrong) {
    wrong = !DecodeBase58(dummy1, result);
    if (result.size() < 25) {
      dummy1.append("1");
      nbDigit++;
    }
  }

  if (wrong) {
    printf("VanitySearch: Wrong character 0, I, O and l not allowed !");
    exit(-1);
  }

  if (result.size() != 25) {
    printf("VanitySearch: Wrong prefix !");
    exit(-1);
  }

  //printf("VanitySearch: Found prefix %s\n",GetHex(result).c_str() );
  sPrefix = *(prefix_t *)(result.data()+1);

  dummy1.append("1");
  DecodeBase58(dummy1, result);

  if (result.size() == 25) {
    //printf("VanitySearch: Found prefix %s\n", GetHex(result).c_str());
    sPrefix = *(prefix_t *)(result.data()+1);
    nbDigit++;
  }
  
  // Difficulty
  _difficulty = pow(2,192) / pow(58,nbDigit);
  printf("Difficulty: %.0f\n", _difficulty);

  // Compute Generator table G[n] = (n+1)*G

  Point g = secp.G;
  Gn[0] = g;
  g = secp.DoubleDirect(g);
  Gn[1] = g;
  for (int i = 2; i < CPU_GRP_SIZE; i++) {
    g = secp.AddDirect(g,secp.G);
    Gn[i] = g;
  }

  // Seed
  if (seed.length() == 0) {
#ifdef WIN64
    // Default seed
    seed = to_string(Timer::qwTicksPerSec.LowPart) + to_string(Timer::perfTickStart.HighPart) +
           to_string(Timer::perfTickStart.LowPart) + to_string(time(NULL));
#else
    // TODO
    seed = to_string(time(NULL));
#endif
  }

  // Protect seed against "seed search attack" using pbkdf2_hmac_sha512
  string salt = "VanitySearch";
  unsigned char hseed[64];
  pbkdf2_hmac_sha512(hseed, 64, (const uint8_t *)seed.c_str(), seed.length(),
    (const uint8_t *)salt.c_str(), salt.length(),
    2048);
  startKey.SetInt32(0);
  sha256(hseed, 64, (unsigned char *)startKey.bits64);

  printf("Base Key:%s\n",startKey.GetBase16().c_str());

}

// ----------------------------------------------------------------------------

string VanitySearch::GetExpectedTime(double keyRate,double keyCount) {

  char tmp[128];
  string ret;

  double P = 1.0/_difficulty;
  // pow(1-P,keyCount) is the probality of failure after keyCount tries
  double cP = 1.0 - pow(1-P,keyCount);

  sprintf(tmp,"[P %.2f%%]",cP*100.0);
  ret = string(tmp);
  
  double desiredP = 0.5;
  while(desiredP<cP)
    desiredP += 0.1;
  if(desiredP>=0.99) desiredP = 0.99;

  double k = log(1.0-desiredP)/log(1.0-P);

  int64_t dTime = (int64_t)((k-keyCount)/keyRate); // Time to perform k tries

  if(dTime<0) dTime = 0;

  double dP = 1.0 - pow(1 - P, k);

  int nbDay  = (int)(dTime / 86400 );
  if (nbDay >= 1) {

    sprintf(tmp, "[%.2f%% in %.1fd]", dP*100.0, (double)dTime / 86400);

  } else {

    int nbHour = (int)((dTime % 86400) / 3600);
    int nbMin = (int)(((dTime % 86400) % 3600) / 60);
    int nbSec = (int)(dTime % 60);

    sprintf(tmp, "[%.2f%% in %02d:%02d:%02d]", dP*100.0, nbHour, nbMin, nbSec);

  }



  return ret + string(tmp);

}

// ----------------------------------------------------------------------------
void VanitySearch::output(string addr,string pAddr,string pAddrHex, string chkAddr,string chkAddrC) {

#ifdef WIN64
   WaitForSingleObject(ghMutex,INFINITE);
#else
  pthread_mutex_lock(&ghMutex);
#endif
  
  FILE *f = stdout;
  bool needToClose = false;

  if (outputFile.length() > 0) {
    f = fopen(outputFile.c_str(), "a");
    if (f == NULL) {
      printf("Cannot open %s for writing\n", outputFile.c_str());
      f = stdout;
    } else {
      needToClose = true;
    }
  }

  fprintf(f, "Pub Addr: %s\n", addr.c_str());
  fprintf(f, "Prv Addr: %s\n", pAddr.c_str());
  fprintf(f, "Prv Key : 0x%s\n", pAddrHex.c_str());
  fprintf(f, "Check   : %s\n", chkAddr.c_str());
  fprintf(f, "Check   : %s (comp)\n\n", chkAddrC.c_str());

  if(needToClose)
    fclose(f);

#ifdef WIN64
  ReleaseMutex(ghMutex);
#else
  pthread_mutex_unlock(&ghMutex);
#endif

}

bool VanitySearch::checkAddr(string &addr, Int &key, uint64_t incr) {

  char p[64];
  char a[64];

  strcpy(p,vanityPrefix.c_str());
  strcpy(a,addr.c_str());
  a[vanityPrefix.length()] = 0;

  if (strcmp(p, a) == 0) {
    // Found it
    Int k(&key);
    k.Add(incr);
    Point p = secp.ComputePublicKey(&k);
    output(addr, secp.GetPrivAddress(k), k.GetBase16(), secp.GetAddress(p, false), secp.GetAddress(p, true));
    nbFoundKey++;
    return true;
  }

  /*
  if (stricmp(p,a) == 0) {
    // Found it (case unsensitive)
    printf("\nFound address:\n");
    printf("Pub Addr: %s\n", addr.c_str());
    printf("Prv Addr: %s\n", secp.GetPrivAddress(key).c_str());
    printf("Prv Key : 0x%s\n", key.GetBase16().c_str());
    //Point p = secp.ComputePublicKey(&key);
    //printf("Check :%s\n", secp.GetAddress(p,true).c_str());
    return true;
  }
  */

  return false;
}

// ----------------------------------------------------------------------------

#ifdef WIN64
DWORD WINAPI _FindKey(LPVOID lpParam) {
#else
void *_FindKey(void *lpParam) {
#endif
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->FindKeyCPU(p);
  return 0;
}

#ifdef WIN64
DWORD WINAPI _FindKeyGPU(LPVOID lpParam) {
#else
void *_FindKeyGPU(void *lpParam) {
#endif
  ((VanitySearch *)lpParam)->FindKeyGPU();
  return 0;
}

// ----------------------------------------------------------------------------

void VanitySearch::FindKeyCPU(TH_PARAM *ph) {

  unsigned char h0[20];
  unsigned char h1[20];
  unsigned char h2[20];
  unsigned char h3[20];

  // Global init
  int thId = ph->threadId;
  counters[thId] = 0;

  // CPU Thread
  IntGroup *grp = new IntGroup();

  // Group Init
  Int key(&startKey);
  Int off((int64_t)0);
  off.Add((uint64_t)thId);
  off.ShiftL(64);
  key.Add(&off);
  Point startP = secp.ComputePublicKey(&key);

  Int dx[CPU_GRP_SIZE];
  Point pts[CPU_GRP_SIZE];
  Int dy;
  Int _s;
  Int _p;
  Point p = startP;
  grp->Set(dx);

  while (!endOfSearch) {

    // Fill group

    for (int i = 0; i < CPU_GRP_SIZE; i++) {
      dx[i].ModSub(&Gn[i].x, &startP.x);
    }

    // Grouped ModInv
    grp->ModInv();

    for (int i = 0; i < CPU_GRP_SIZE; i++) {

      pts[i] = p;
      p = startP;

      dy.ModSub(&Gn[i].y, &p.y);

      _s.MontgomeryMult(&dy, &dx[i]);     // s = (p2.y-p1.y)*inverse(p2.x-p1.x);
      _p.MontgomeryMult(&_s, &_s);        // _p = pow2(s)*R^-3
      _p.MontgomeryMult(Int::GetR4());    // _p = pow2(s)

      p.x.ModNeg();
      p.x.ModAdd(&_p);
      p.x.ModSub(&Gn[i].x);               // rx = pow2(s) - p1.x - p2.x;

      p.y.ModSub(&Gn[i].x, &p.x);
      p.y.MontgomeryMult(&_s);
      p.y.MontgomeryMult(Int::GetR3());
      p.y.ModSub(&Gn[i].y);               // ry = - p2.y - s*(ret.x-p2.x);  

    }

    // Check addresses (compressed)

    for (int i = 0; i < CPU_GRP_SIZE; i += 4) {

      secp.GetHash160(searchComp, pts[i], pts[i+1], pts[i+2], pts[i+3], h0, h1, h2, h3);

      prefix_t pr0 = *(prefix_t *)h0;
      prefix_t pr1 = *(prefix_t *)h1;
      prefix_t pr2 = *(prefix_t *)h2;
      prefix_t pr3 = *(prefix_t *)h3;

      if (pr0 == sPrefix) {
        string addr = secp.GetAddress(pts[i], searchComp);
        endOfSearch = checkAddr(addr, key, i) && stopWhenFound;
      }
      if (pr1 == sPrefix) {
        string addr = secp.GetAddress(pts[i+1], searchComp);
        endOfSearch = checkAddr(addr, key, i + 1) && stopWhenFound;
      }
      if (pr2 == sPrefix) {
        string addr = secp.GetAddress(pts[i+2], searchComp);
        endOfSearch = checkAddr(addr, key, i + 2) && stopWhenFound;
      }
      if (pr3 == sPrefix) {
        string addr = secp.GetAddress(pts[i+3], searchComp);
        endOfSearch = checkAddr(addr, key, i + 3) && stopWhenFound;
      }

    }

    key.Add((uint64_t)CPU_GRP_SIZE);
    startP = p;
    counters[thId]+= CPU_GRP_SIZE;

  }


}

// ----------------------------------------------------------------------------

void VanitySearch::FindKeyGPU() {

  bool ok = true;

#ifdef WITHGPU

  // Global init
  GPUEngine g(gridSize, gpuId);
  int nbThread = g.GetNbThread();
  Point *p = new Point[nbThread];
  Int *keys = new Int[nbThread];
  vector<ITEM> found;

  printf("GPU: %s\n",g.deviceName.c_str());

  counters[0xFF] = 0;

  for (int i = 0; i < nbThread; i++) {
    keys[i].Set(&startKey);
    Int off((uint64_t)i);
    off.ShiftL(96);
    keys[i].Add(&off);
    p[i] = secp.ComputePublicKey(&keys[i]);
  }
  g.SetSearchMode(searchComp);
  g.SetPrefix(sPrefix);
  ok = g.SetKeys(p);

  // GPU Thread
  while (ok && !endOfSearch) {

    // Call kernel
    ok = g.Launch(found);

    for(int i=0;i<(int)found.size() && !endOfSearch;i++) {

      ITEM it = found[i];
      string addr = secp.GetAddress(it.hash, searchComp);
      endOfSearch = checkAddr(addr, keys[it.thId], it.incr) && stopWhenFound;
 
    }

    if (ok) {
      for (int i = 0; i < nbThread; i++) {
        keys[i].Add((uint64_t)STEP_SIZE);
      }
      counters[0xFF] += STEP_SIZE * nbThread;
    }

  }

  // GPU thread may exit on error
  if(nbCpuThread==0)
    endOfSearch = true;

  delete[] keys;
  delete[] p;

#else
  printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif

}

// ----------------------------------------------------------------------------

void VanitySearch::Search(int nbThread) {

  double t0;
  double t1;
  endOfSearch = false;
  nbCpuThread = nbThread;
  nbFoundKey = 0;

  memset(counters,0,sizeof(counters));

  printf("Number of CPU thread: %d\n", nbThread);

  TH_PARAM *params = (TH_PARAM *)malloc((nbThread + (useGpu?1:0)) * sizeof(TH_PARAM));

  for (int i = 0; i < nbThread; i++) {
    params[i].obj = this;
    params[i].threadId = i;

#ifdef WIN64
    DWORD thread_id;
    CreateThread(NULL, 0, _FindKey, (void*)(params+i), 0, &thread_id);
    ghMutex = CreateMutex(NULL, FALSE, NULL);
#else
    pthread_t thread_id;
    pthread_create(&thread_id, NULL, &_FindKey, (void*)(params+i));  
    ghMutex = PTHREAD_MUTEX_INITIALIZER;
#endif
  }

  if (useGpu) {
    params[nbThread].obj = this;
    params[nbThread].threadId = 255;
#ifdef WIN64
    DWORD thread_id;
    CreateThread(NULL, 0, _FindKeyGPU, (void*)(this), 0, &thread_id);
#else
    pthread_t thread_id;
    pthread_create(&thread_id, NULL, &_FindKeyGPU, (void*)(this));  
#endif
  }

#ifndef WIN64
  setvbuf(stdout, NULL, _IONBF, 0);
#endif

  t0 = Timer::get_tick();
  startTime = t0;
  uint64_t lastCount = 0;
  uint64_t lastGPUCount = 0;
  while (!endOfSearch) {

    int delay = 2000;
    while (!endOfSearch && delay>0) {
#ifdef WIN64
      Sleep(500);
#else
      usleep(500000);
#endif
      delay -= 500;
    }

    uint64_t count = 0;
    for (int i = 0; i < nbThread; i++)
      count += counters[i];
    if(useGpu)
      count += counters[0xFF];

    t1 = Timer::get_tick();
    double keyRate = (double)(count - lastCount) / (t1 - t0);
    double gpuKeyRate = (double)(counters[0xFF] - lastGPUCount) / (t1 - t0);

    if (!endOfSearch) {
      if (stopWhenFound) {
        printf("%.3f MK/s (GPU %.3f MK/s) (2^%.2f) %s\r",
          keyRate / 1000000.0, gpuKeyRate / 1000000.0,
          log2((double)count), GetExpectedTime(keyRate, (double)count).c_str());
      } else {
        printf("%.3f MK/s (GPU %.3f MK/s) (2^%.2f) %s[%d]\r",
          keyRate / 1000000.0, gpuKeyRate / 1000000.0,
          log2((double)count), GetExpectedTime(keyRate, (double)count).c_str(),nbFoundKey);
      }
    }

    lastCount = count;
    lastGPUCount = counters[0xFF];
    t0 = t1;

  }

  free(params);

}

// ----------------------------------------------------------------------------

string VanitySearch::GetHex(vector<unsigned char> &buffer) {

  string ret;

  char tmp[128];
  for (int i = 0; i < (int)buffer.size(); i++) {
    sprintf(tmp,"%02X",buffer[i]);
    ret.append(tmp);
  }

  return ret;

}
