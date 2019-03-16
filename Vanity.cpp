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

#include "Vanity.h"
#include "Base58.h"
#include "hash/sha256.h"
#include "hash/sha512.h"
#include "IntGroup.h"
#include "Timer.h"
#include "hash/ripemd160.h"
#include <string.h>
#include <math.h>
#include <algorithm>
#ifndef WIN64
#include <pthread.h>
#include <unistd.h>
#endif

using namespace std;

Point Gn[CPU_GRP_SIZE / 2];
Point _2Gn;

// ----------------------------------------------------------------------------

VanitySearch::VanitySearch(Secp256K1 &secp, vector<std::string> prefix,string seed,bool comp, 
                           bool useGpu, bool stop, string outputFile, bool useSSE) {

  this->secp = secp;
  this->searchComp = comp;
  this->useGpu = useGpu;
  this->stopWhenFound = stop;
  this->outputFile = outputFile;
  this->useSSE = useSSE;
  this->nbGPUThread = 0;
  prefixes.clear();

  char *ctimeBuff;
  time_t now = time(NULL);
  ctimeBuff = ctime(&now);
  printf("Start %s", ctimeBuff);

  // Create a 65536 items lookup table
  PREFIX_TABLE_ITEM t;
  t.found = true;
  t.items = NULL;
  for(int i=0;i<65536;i++)
    prefixes.push_back(t);

  // Insert prefixes
  nbPrefix = 0;
  onlyFull = true;
  for (int i = 0; i < (int)prefix.size(); i++) {
    PREFIX_ITEM it;
    if (initPrefix(prefix[i], &it)) {
      std::vector<PREFIX_ITEM> *items;
      items = prefixes[it.sPrefix].items;
      if (items == NULL) {
        prefixes[it.sPrefix].items = new vector<PREFIX_ITEM>();
        prefixes[it.sPrefix].found = false;
        items = prefixes[it.sPrefix].items;
        usedPrefix.push_back(it.sPrefix);
      }
      items->push_back(it);
      onlyFull &= it.isFull;
      nbPrefix++;
    }
  }
  //dumpPrefixes();

  if (nbPrefix == 0) {
    printf("VanitySearch: nothing to search !\n");
    exit(1);
  }

  // Second level lookup
  uint16_t unique_sPrefix = 0;
  for (int i = 0; i < prefixes.size(); i++) {
    std::vector<PREFIX_ITEM> *items;
    items = prefixes[i].items;
    if (items) {
      LPREFIX lit;
      lit.sPrefix = i;
      for (int j = 0; j < items->size(); j++)
        lit.lPrefixes.push_back((*items)[j].lPrefix);
      sort(lit.lPrefixes.begin(), lit.lPrefixes.end());
      usedPrefixL.push_back(lit);
      unique_sPrefix++;
    }
  }

  _difficulty = getDiffuclty();
  if (nbPrefix == 1) {
    prefix_t p0 = usedPrefix[0];
    printf("Difficulty: %.0f\n", _difficulty);
    printf("Search: %s\n", (*prefixes[p0].items)[0].prefix.c_str());
  } else {
    printf("Search: %d prefixes (Lookup size %d)\n", nbPrefix, unique_sPrefix);
  }

  // Compute Generator table G[n] = (n+1)*G

  Point g = secp.G;
  Gn[0] = g;
  g = secp.DoubleDirect(g);
  Gn[1] = g;
  for (int i = 2; i < CPU_GRP_SIZE/2; i++) {
    g = secp.AddDirect(g,secp.G);
    Gn[i] = g;
  }
  // _2Gn = CPU_GRP_SIZE*G
  _2Gn = secp.DoubleDirect(Gn[CPU_GRP_SIZE/2-1]);

  // Constant for endomorphism
  // (beta^3 = 1 mod p) 
  // (lambda^3 = 1 mod n)
  beta.SetBase16("7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee");
  lambda.SetBase16("5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72");
  beta2.SetBase16("851695d49a83f8ef919bb86153cbcb16630fb68aed0a766a3ec693d68e6afa40");
  lambda2.SetBase16("ac9c52b33fa3cf1f5ad9e3fd77ed9ba4a880b9fc8ec739c2e0cfc810b51283ce");

  // Seed
  if (seed.length() == 0) {
    // Default seed
    seed = to_string(Timer::getSeedFromTimer());
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
bool VanitySearch::initPrefix(std::string prefix,PREFIX_ITEM *it) {

  std::vector<unsigned char> result;
  string dummy1 = prefix;
  int nbDigit = 0;
  bool wrong = false;

  if (prefix.length() < 2) {
    printf("Ignoring prefix \"%s\" (too short)\n",prefix.c_str());
    return false;
  }

  if (prefix.data()[0] != '1') {
    printf("Ignoring prefix \"%s\" (must start with 1)\n", prefix.c_str());
    return false;
  }

  // Search for highest hash160 16bit prefix (most probable)

  while (result.size() < 25 && !wrong) {
    wrong = !DecodeBase58(dummy1, result);
    if (result.size() < 25) {
      dummy1.append("1");
      nbDigit++;
    }
  }

  if (wrong) {
    printf("Ignoring prefix \"%s\" (0, I, O and l not allowed)\n", prefix.c_str());
    return false;
  }

  if (result.size() != 25) {
    printf("Ignoring prefix \"%s\" (Invalid size)\n", prefix.c_str());
    return false;
  }

  //printf("VanitySearch: Found prefix %s\n",GetHex(result).c_str() );
  it->sPrefix = *(prefix_t *)(result.data() + 1);

  dummy1.append("1");
  DecodeBase58(dummy1, result);

  if (result.size() == 25) {
    //printf("VanitySearch: Found prefix %s\n", GetHex(result).c_str());
    it->sPrefix = *(prefix_t *)(result.data() + 1);
    nbDigit++;
  }

  // Try to attack a full address ?
  DecodeBase58(prefix, result);
  if (result.size() > 21) {

    // mamma mia !
    if (!secp.CheckPudAddress(prefix)) {
      printf("Warning, \"%s\" (address checksum may never match)\n", prefix.c_str());
    }
    it->difficulty = pow(2, 160);
    it->isFull = true;
    memcpy(it->hash160, result.data() + 1, 20);
    it->lPrefix = *(prefixl_t *)(it->hash160);

  } else {

    // Difficulty
    it->difficulty = pow(2, 192) / pow(58, nbDigit);
    it->isFull = false;
    it->lPrefix = 0;

  }

  it->prefix = prefix;
  it->found = false;

  return true;

}

// ----------------------------------------------------------------------------

void VanitySearch::dumpPrefixes() {

  for (int i = 0; i < 0xFFFF; i++) {
    if (prefixes[i].items) {
      printf("%04X\n", i);
      std::vector<PREFIX_ITEM> *items = prefixes[i].items;
      for (int j = 0; j < (*items).size(); j++) {
        printf("  %d\n", (*items)[j].sPrefix);
        printf("  %g\n", (*items)[j].difficulty);
        printf("  %s\n", (*items)[j].prefix.c_str());
      }
    }
  }

}

// ----------------------------------------------------------------------------

double VanitySearch::getDiffuclty() {

  double min = pow(2,160);

  if (onlyFull)
    return min;

  for (int i = 0; i < (int)usedPrefix.size(); i++) {
    int p = usedPrefix[i];
    std::vector<PREFIX_ITEM>& items = *prefixes[p].items;
    for (int j = 0; j < items.size(); j++) {
      if (!items[j].found) {
        if(items[j].difficulty<min)
          min = items[j].difficulty;
      }
    }
  }

  return min;

}

double log1(double x) {
  // Use taylor series to approximate log(1-x)
  return -x - (x*x)/2.0 - (x*x*x)/3.0 - (x*x*x*x)/4.0;
}

string VanitySearch::GetExpectedTime(double keyRate,double keyCount) {

  char tmp[128];
  string ret;

  double P = 1.0/ _difficulty;
  // pow(1-P,keyCount) is the probality of failure after keyCount tries
  double cP = 1.0 - pow(1-P,keyCount);

  sprintf(tmp,"[P %.2f%%]",cP*100.0);
  ret = string(tmp);
  
  double desiredP = 0.5;
  while(desiredP<cP)
    desiredP += 0.1;
  if(desiredP>=0.99) desiredP = 0.99;
  double k = log(1.0-desiredP)/log(1.0-P);
  if (isinf(k)) {
    // Try taylor
    k = log(1.0 - desiredP)/log1(P);
  }
  double dTime = (k-keyCount)/keyRate; // Time to perform k tries

  if(dTime<0) dTime = 0;

  double nbDay  = dTime / 86400.0;
  if (nbDay >= 1) {

    double nbYear = nbDay/365.0;
    if (nbYear > 1) {
      if(nbYear<5)
        sprintf(tmp, "[%.2f%% in %.1fy]", desiredP*100.0, nbYear);
      else
        sprintf(tmp, "[%.2f%% in %gy]", desiredP*100.0, nbYear);
    } else {
      sprintf(tmp, "[%.2f%% in %.1fd]", desiredP*100.0, nbDay);
    }

  } else {

    int iTime = (int)dTime;
    int nbHour = (int)((iTime % 86400) / 3600);
    int nbMin = (int)(((iTime % 86400) % 3600) / 60);
    int nbSec = (int)(iTime % 60);

    sprintf(tmp, "[%.2f%% in %02d:%02d:%02d]", desiredP*100.0, nbHour, nbMin, nbSec);

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

  fprintf(f, "\nPub Addr: %s\n", addr.c_str());
  fprintf(f, "Prv Addr: %s\n", pAddr.c_str());
  fprintf(f, "Prv Key : 0x%s\n", pAddrHex.c_str());
  fprintf(f, "Check   : %s\n", chkAddr.c_str());
  fprintf(f, "Check   : %s (comp)\n", chkAddrC.c_str());

  if(needToClose)
    fclose(f);

#ifdef WIN64
  ReleaseMutex(ghMutex);
#else
  pthread_mutex_unlock(&ghMutex);
#endif

}

// ----------------------------------------------------------------------------

void VanitySearch::updateFound() {

  // Check if all prefixes has been found
  // Needed only if stopWhenFound is asked
  if (stopWhenFound) {

    bool allFound = true;
    for (int i = 0; i < usedPrefix.size(); i++) {
      bool iFound = true;
      if (!prefixes[usedPrefix[i]].found) {
        std::vector<PREFIX_ITEM>& items = *prefixes[usedPrefix[i]].items;
        for (int j = 0; j < items.size(); j++)
          iFound &= items[j].found;
        prefixes[usedPrefix[i]].found = iFound;
      }
      allFound &= iFound;
    }
    endOfSearch = allFound;

    // Update difficulty to the next most probable item
    _difficulty = getDiffuclty();

  }

}

// ----------------------------------------------------------------------------

void VanitySearch::checkAddr(int prefIdx, uint8_t *hash160, Int &key, int32_t incr, int endomorphism) {

  vector<PREFIX_ITEM>& items = *prefixes[prefIdx].items;

  for (int i = 0; i < items.size(); i++) {

    if(stopWhenFound && items[i].found)
      continue;

    if (items[i].isFull) {

      if (ripemd160_comp_hash(items[i].hash160, hash160) ) {

        // Found it !
        // You believe it ?
        // Mark it as found
        items[i].found = true;
        Int k(&key);
        if(incr<0) {
          k.Add((uint64_t)(-incr));
          k.Neg();
          k.Add(&secp.order);
        } else {
          k.Add((uint64_t)incr);
        }

        // Endomorphisms
        switch (endomorphism) {
          case 1:
            k.ModMulK1order(&lambda);
            break;
          case 2:
            k.ModMulK1order(&lambda2);
            break;
        }

        Point p = secp.ComputePublicKey(&k);
        string addr = secp.GetAddress(hash160, searchComp);
        output(addr, secp.GetPrivAddress(k), k.GetBase16(), secp.GetAddress(p, false), secp.GetAddress(p, true));
        nbFoundKey++;
        updateFound();
      }

    } else {

      char p[64];
      char a[64];

      strcpy(p, items[i].prefix.c_str());
      string addr = secp.GetAddress(hash160,searchComp);
      strcpy(a, addr.c_str());
      a[items[i].prefix.length()] = 0;

      if (strcmp(p, a) == 0) {

        // Mark it as found
        items[i].found = true;
        Int k(&key);
        if (incr < 0) {
          k.Add((uint64_t)(-incr));
          k.Neg();
          k.Add(&secp.order);
        } else {
          k.Add((uint64_t)incr);
        }
        // Endomorphisms
        switch (endomorphism) {
          case 1:
            k.ModMulK1order(&lambda);
            break;
          case 2:
            k.ModMulK1order(&lambda2);
            break;
        }
        printf("\nFound: incr=%d endo=%d\n", incr, endomorphism);
        Point p = secp.ComputePublicKey(&k);
        output(addr, secp.GetPrivAddress(k), k.GetBase16(), secp.GetAddress(p, false), secp.GetAddress(p, true));
        nbFoundKey++;
        updateFound();
      }

    }

  }

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
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->FindKeyGPU(p);
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
  IntGroup *grp = new IntGroup(CPU_GRP_SIZE/2+1);

  // Group Init
  Int key(&startKey);
  Int off((int64_t)thId);
  off.ShiftL(64);
  key.Add(&off);
  Int km(&key);
  km.Add((uint64_t)CPU_GRP_SIZE/2);
  Point startP = secp.ComputePublicKey(&km);

  Int dx[CPU_GRP_SIZE/2+1];
  Point pts[CPU_GRP_SIZE];
  Point pte1[4];
  Point pte2[4];

  Int dy;
  Int dyn;
  Int _s;
  Int _p;
  Point pp;
  Point pn;
  grp->Set(dx);

  while (!endOfSearch) {

    // Fill group
    int i;
    int hLength = (CPU_GRP_SIZE / 2 - 1);

    for (i = 0; i < hLength; i++) {
      dx[i].ModSub(&Gn[i].x, &startP.x);
    }
    dx[i].ModSub(&Gn[i].x, &startP.x);  // For the first point
    dx[i+1].ModSub(&_2Gn.x, &startP.x); // For the next center point

    // Grouped ModInv
    grp->ModInv();

    // We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
    // We compute key in the positive and negative way from the center of the group

    // center point
    pts[CPU_GRP_SIZE/2] = startP;

    for (i = 0; i<hLength && !endOfSearch; i++) {

      pp = startP;
      pn = startP;

      // P = startP + i*G
      dy.ModSub(&Gn[i].y,&pp.y);

      _s.ModMulK1(&dy, &dx[i]);       // s = (p2.y-p1.y)*inverse(p2.x-p1.x);
      _p.ModSquareK1(&_s);            // _p = pow2(s)

      pp.x.ModNeg();
      pp.x.ModAdd(&_p);
      pp.x.ModSub(&Gn[i].x);           // rx = pow2(s) - p1.x - p2.x;

      pp.y.ModSub(&Gn[i].x, &pp.x);
      pp.y.ModMulK1(&_s);
      pp.y.ModSub(&Gn[i].y);           // ry = - p2.y - s*(ret.x-p2.x);  

      // P = startP - i*G  , if (x,y) = i*G then (x,-y) = -i*G
      dyn.Set(&Gn[i].y);
      dyn.ModNeg();
      dyn.ModSub(&pn.y);

      _s.ModMulK1(&dyn, &dx[i]);      // s = (p2.y-p1.y)*inverse(p2.x-p1.x);
      _p.ModSquareK1(&_s);            // _p = pow2(s)

      pn.x.ModNeg();
      pn.x.ModAdd(&_p);
      pn.x.ModSub(&Gn[i].x);          // rx = pow2(s) - p1.x - p2.x;

      pn.y.ModSub(&Gn[i].x, &pn.x);
      pn.y.ModMulK1(&_s);
      pn.y.ModAdd(&Gn[i].y);          // ry = - p2.y - s*(ret.x-p2.x);  

      pts[CPU_GRP_SIZE/2 + (i+1)] = pp;
      pts[CPU_GRP_SIZE/2 - (i+1)] = pn;

    }

    // First point (startP - (GRP_SZIE/2)*G)
    pn = startP;
    dyn.Set(&Gn[i].y);
    dyn.ModNeg();
    dyn.ModSub(&pn.y);

    _s.ModMulK1(&dyn, &dx[i]);
    _p.ModSquareK1(&_s);

    pn.x.ModNeg();
    pn.x.ModAdd(&_p);
    pn.x.ModSub(&Gn[i].x);

    pn.y.ModSub(&Gn[i].x, &pn.x);
    pn.y.ModMulK1(&_s);
    pn.y.ModAdd(&Gn[i].y);

    pts[0] = pn;

    // Next start point (startP + GRP_SIZE*G)
    pp = startP;
    dy.ModSub(&_2Gn.y, &pp.y);

    _s.ModMulK1(&dy, &dx[i+1]);
    _p.ModSquareK1(&_s);

    pp.x.ModNeg();
    pp.x.ModAdd(&_p);
    pp.x.ModSub(&_2Gn.x);

    pp.y.ModSub(&_2Gn.x, &pp.x);
    pp.y.ModMulK1(&_s);
    pp.y.ModSub(&_2Gn.y);
    startP = pp;

#if 0
    // Check
    {
      bool wrong = false;
      Point p0 = secp.ComputePublicKey(&key);
      for (int i = 0; i < CPU_GRP_SIZE; i++) {
        if (!p0.equals(pts[i])) {
          wrong = true;
          printf("[%d] wrong point\n",i);
        }
        p0 = secp.NextKey(p0);
      }
      if(wrong) exit(0);
    }
#endif

    // Check addresses
    if (useSSE) {

      for (int i = 0; i < CPU_GRP_SIZE && !endOfSearch; i += 4) {

        // Point -------------------------------------------------------------------------
        secp.GetHash160(searchComp, pts[i], pts[i + 1], pts[i + 2], pts[i + 3], h0, h1, h2, h3);

        prefix_t pr0 = *(prefix_t *)h0;
        prefix_t pr1 = *(prefix_t *)h1;
        prefix_t pr2 = *(prefix_t *)h2;
        prefix_t pr3 = *(prefix_t *)h3;

        if (prefixes[pr0].items)
           checkAddr(pr0, h0, key, i, 0);
        if (prefixes[pr1].items)
           checkAddr(pr1, h1, key, i+1, 0);
        if (prefixes[pr2].items)
           checkAddr(pr2, h2, key, i+2, 0);
        if (prefixes[pr3].items)
           checkAddr(pr3, h3, key, i+3, 0);

        // Endomorphism #1
        // if (x, y) = k * G, then (beta*x, y) = lambda*k*G
        pte1[0].x.ModMulK1(&pts[i].x,&beta);
        pte1[0].y.Set(&pts[i].y);
        pte1[1].x.ModMulK1(&pts[i+1].x, &beta);
        pte1[1].y.Set(&pts[i+1].y);
        pte1[2].x.ModMulK1(&pts[i + 2].x, &beta);
        pte1[2].y.Set(&pts[i + 2].y);
        pte1[3].x.ModMulK1(&pts[i + 3].x, &beta);
        pte1[3].y.Set(&pts[i + 3].y);

        secp.GetHash160(searchComp, pte1[0], pte1[1], pte1[2], pte1[3], h0, h1, h2, h3);

        pr0 = *(prefix_t *)h0;
        pr1 = *(prefix_t *)h1;
        pr2 = *(prefix_t *)h2;
        pr3 = *(prefix_t *)h3;

        if (prefixes[pr0].items)
          checkAddr(pr0, h0, key, i, 1);
        if (prefixes[pr1].items)
          checkAddr(pr1, h1, key, (i + 1), 1);
        if (prefixes[pr2].items)
          checkAddr(pr2, h2, key, (i + 2), 1);
        if (prefixes[pr3].items)
          checkAddr(pr3, h3, key, (i + 3), 1);

        // Endomorphism #2
        // if (x, y) = k * G, then (beta2*x, y) = lambda2*k*G
        pte2[0].x.ModMulK1(&pts[i].x, &beta2);
        pte2[0].y.Set(&pts[i].y);
        pte2[1].x.ModMulK1(&pts[i + 1].x, &beta2);
        pte2[1].y.Set(&pts[i + 1].y);
        pte2[2].x.ModMulK1(&pts[i + 2].x, &beta2);
        pte2[2].y.Set(&pts[i + 2].y);
        pte2[3].x.ModMulK1(&pts[i + 3].x, &beta2);
        pte2[3].y.Set(&pts[i + 3].y);

        secp.GetHash160(searchComp, pte2[0], pte2[1], pte2[2], pte2[3], h0, h1, h2, h3);

        pr0 = *(prefix_t *)h0;
        pr1 = *(prefix_t *)h1;
        pr2 = *(prefix_t *)h2;
        pr3 = *(prefix_t *)h3;

        if (prefixes[pr0].items)
          checkAddr(pr0, h0, key, i, 2);
        if (prefixes[pr1].items)
          checkAddr(pr1, h1, key, (i + 1), 2);
        if (prefixes[pr2].items)
          checkAddr(pr2, h2, key, (i + 2), 2);
        if (prefixes[pr3].items)
          checkAddr(pr3, h3, key, (i + 3), 2);

        // Curve symetrie -------------------------------------------------------------------------
        // if (x,y) = k*G, then (x, -y) is -k*G

        pts[i].y.ModNeg();
        pts[i+1].y.ModNeg();
        pts[i+2].y.ModNeg();
        pts[i+3].y.ModNeg();

        secp.GetHash160(searchComp, pts[i], pts[i + 1], pts[i + 2], pts[i + 3], h0, h1, h2, h3);

        pr0 = *(prefix_t *)h0;
        pr1 = *(prefix_t *)h1;
        pr2 = *(prefix_t *)h2;
        pr3 = *(prefix_t *)h3;

        if (prefixes[pr0].items)
          checkAddr(pr0, h0, key, -i, 0);
        if (prefixes[pr1].items)
          checkAddr(pr1, h1, key, -(i + 1), 0);
        if (prefixes[pr2].items)
          checkAddr(pr2, h2, key, -(i + 2), 0);
        if (prefixes[pr3].items)
          checkAddr(pr3, h3, key, -(i + 3), 0);

        // Endomorphism #1
        // if (x, y) = k * G, then (beta*x, y) = lambda*k*G
        pte1[0].y.ModNeg();
        pte1[1].y.ModNeg();
        pte1[2].y.ModNeg();
        pte1[3].y.ModNeg();


        secp.GetHash160(searchComp, pte1[0], pte1[1], pte1[2], pte1[3], h0, h1, h2, h3);

        pr0 = *(prefix_t *)h0;
        pr1 = *(prefix_t *)h1;
        pr2 = *(prefix_t *)h2;
        pr3 = *(prefix_t *)h3;

        if (prefixes[pr0].items)
          checkAddr(pr0, h0, key, -i, 1);
        if (prefixes[pr1].items)
          checkAddr(pr1, h1, key, -(i + 1), 1);
        if (prefixes[pr2].items)
          checkAddr(pr2, h2, key, -(i + 2), 1);
        if (prefixes[pr3].items)
          checkAddr(pr3, h3, key, -(i + 3), 1);

        // Endomorphism #2
        // if (x, y) = k * G, then (beta2*x, y) = lambda2*k*G
        pte2[0].y.ModNeg();
        pte2[1].y.ModNeg();
        pte2[2].y.ModNeg();
        pte2[3].y.ModNeg();

        secp.GetHash160(searchComp, pte2[0], pte2[1], pte2[2], pte2[3], h0, h1, h2, h3);

        pr0 = *(prefix_t *)h0;
        pr1 = *(prefix_t *)h1;
        pr2 = *(prefix_t *)h2;
        pr3 = *(prefix_t *)h3;

        if (prefixes[pr0].items)
          checkAddr(pr0, h0, key, -i, 2);
        if (prefixes[pr1].items)
          checkAddr(pr1, h1, key, -(i + 1), 2);
        if (prefixes[pr2].items)
          checkAddr(pr2, h2, key, -(i + 2), 2);
        if (prefixes[pr3].items)
          checkAddr(pr3, h3, key, -(i + 3), 2);


      }

    } else {

      for (int i = 0; i < CPU_GRP_SIZE && !endOfSearch; i ++) {


        // Point
        secp.GetHash160(pts[i], searchComp, h0);
        prefix_t pr0 = *(prefix_t *)h0;
        if (prefixes[pr0].items)
          checkAddr(pr0,h0, key, i, 0);

        // Endomorphism #1
        pte1[0].x.ModMulK1(&pts[i].x, &beta);
        pte1[0].y.Set(&pts[i].y);

        secp.GetHash160(pte1[0], searchComp, h0);

        pr0 = *(prefix_t *)h0;
        if (prefixes[pr0].items)
          checkAddr(pr0, h0, key, i, 1);

        // Endomorphism #2
        pte2[0].x.ModMulK1(&pts[i].x, &beta2);
        pte2[0].y.Set(&pts[i].y);

        secp.GetHash160(pte2[0], searchComp, h0);

        pr0 = *(prefix_t *)h0;
        if (prefixes[pr0].items)
          checkAddr(pr0, h0, key, i, 2);

        // Curve symetrie
        // if (x,y) = k*G, then (x, -y) is -k*G
        pts[i].y.ModNeg();
        secp.GetHash160(pts[i], searchComp, h0);
        pr0 = *(prefix_t *)h0;
        if (prefixes[pr0].items)
          checkAddr(pr0, h0, key, -i, 0);

        // Endomorphism #1
        pte1[0].y.ModNeg();

        secp.GetHash160(pte1[0], searchComp, h0);

        pr0 = *(prefix_t *)h0;
        if (prefixes[pr0].items)
          checkAddr(pr0, h0, key, -i, 1);

        // Endomorphism #2
        pte2[0].y.ModNeg();

        secp.GetHash160(pte2[0], searchComp, h0);

        pr0 = *(prefix_t *)h0;
        if (prefixes[pr0].items)
          checkAddr(pr0, h0, key, -i, 2);

      }

    }

    key.Add((uint64_t)CPU_GRP_SIZE);
    counters[thId]+= 6*CPU_GRP_SIZE; // Point + endo #1 + endo #2 + Symetric point + endo #1 + endo #2

  }

  ph->isRunning = false;

}

// ----------------------------------------------------------------------------

void VanitySearch::FindKeyGPU(TH_PARAM *ph) {

  bool ok = true;

#ifdef WITHGPU

  // Global init
  int thId = ph->threadId;
  GPUEngine g(ph->gridSize, ph->gpuId);
  int nbThread = g.GetNbThread();
  Point *p = new Point[nbThread];
  Int *keys = new Int[nbThread];
  vector<ITEM> found;

  printf("GPU: %s\n",g.deviceName.c_str());

  counters[thId] = 0;

  for (int i = 0; i < nbThread; i++) {
    keys[i].Set(&startKey);
    Int offT((uint64_t)i);
    offT.ShiftL(80);
    Int offG((uint64_t)thId);
    offG.ShiftL(112);
    keys[i].Add(&offT);
    keys[i].Add(&offG);
    Int k(keys+i);
    // Starting key is at the middle of the group
    k.Add((uint64_t)(g.GetGroupSize()/2));
    p[i] = secp.ComputePublicKey(&k);
  }
  g.SetSearchMode(searchComp);
  if (onlyFull) {
    g.SetPrefix(usedPrefixL,nbPrefix);
  } else {
    g.SetPrefix(usedPrefix);
  }
  ok = g.SetKeys(p);

  // GPU Thread
  while (ok && !endOfSearch) {

    // Call kernel
    ok = g.Launch(found);

    for(int i=0;i<(int)found.size() && !endOfSearch;i++) {

      ITEM it = found[i];
      checkAddr(*(prefix_t *)(it.hash), it.hash, keys[it.thId], it.incr, it.endo);
 
    }

    if (ok) {
      for (int i = 0; i < nbThread; i++) {
        keys[i].Add((uint64_t)STEP_SIZE);
      }
      counters[thId] += 6 * STEP_SIZE * nbThread; // Point +  endo1 + endo2 + symetrics
    }

  }

  delete[] keys;
  delete[] p;

#else
  printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif

  ph->isRunning = false;

}
// ----------------------------------------------------------------------------

bool VanitySearch::isAlive(TH_PARAM *p) {

  bool isAlive = true;
  int total = nbCPUThread + nbGPUThread;
  for(int i=0;i<total;i++)
    isAlive = isAlive && p[i].isRunning;

  return isAlive;

}

uint64_t VanitySearch::getGPUCount() {

  uint64_t count = 0;
  for(int i=0;i<nbGPUThread;i++)
    count += counters[0x80L+i];
  return count;

}

uint64_t VanitySearch::getCPUCount() {

  uint64_t count = 0;
  for(int i=0;i<nbCPUThread;i++)
    count += counters[i];
  return count;

}

// ----------------------------------------------------------------------------

void VanitySearch::Search(int nbThread,std::vector<int> gpuId,std::vector<int> gridSize) {

  double t0;
  double t1;
  endOfSearch = false;
  nbCPUThread = nbThread;
  nbGPUThread = (useGpu?(int)gpuId.size():0);
  nbFoundKey = 0;

  memset(counters,0,sizeof(counters));

  printf("Number of CPU thread: %d\n", nbCPUThread);

  TH_PARAM *params = (TH_PARAM *)malloc((nbCPUThread + nbGPUThread) * sizeof(TH_PARAM));
  memset(params,0,(nbCPUThread + nbGPUThread) * sizeof(TH_PARAM));

  // Launch CPU threads
  for (int i = 0; i < nbCPUThread; i++) {
    params[i].obj = this;
    params[i].threadId = i;
    params[i].isRunning = true;

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

  // Launch GPU threads
  for (int i = 0; i < nbGPUThread; i++) {
    params[nbCPUThread+i].obj = this;
    params[nbCPUThread+i].threadId = 0x80L+i;
    params[nbCPUThread+i].isRunning = true;
    params[nbCPUThread+i].gpuId = gpuId[i];
    params[nbCPUThread+i].gridSize = gridSize[i];
#ifdef WIN64
    DWORD thread_id;
    CreateThread(NULL, 0, _FindKeyGPU, (void*)(params+(nbCPUThread+i)), 0, &thread_id);
#else
    pthread_t thread_id;
    pthread_create(&thread_id, NULL, &_FindKeyGPU, (void*)(params+(nbCPUThread+i)));  
#endif
  }

#ifndef WIN64
  setvbuf(stdout, NULL, _IONBF, 0);
#endif

  t0 = Timer::get_tick();
  startTime = t0;
  uint64_t lastCount = 0;
  uint64_t gpuCount = 0;
  uint64_t lastGPUCount = 0;

  // Key rate smoothing filter
  #define FILTER_SIZE 8
  double lastkeyRate[FILTER_SIZE];
  double lastGpukeyRate[FILTER_SIZE];
  uint32_t filterPos = 0;

  double keyRate = 0.0;
  double gpuKeyRate = 0.0;

  memset(lastkeyRate,0,sizeof(lastkeyRate));
  memset(lastGpukeyRate,0,sizeof(lastkeyRate));

  while (isAlive(params)) {

    int delay = 2000;
    while (isAlive(params) && delay>0) {
#ifdef WIN64
      Sleep(500);
#else
      usleep(500000);
#endif
      delay -= 500;
    }

    gpuCount = getGPUCount();
    uint64_t count = getCPUCount() + gpuCount;

    t1 = Timer::get_tick();
    keyRate = (double)(count - lastCount) / (t1 - t0);
    gpuKeyRate = (double)(gpuCount - lastGPUCount) / (t1 - t0);
    lastkeyRate[filterPos%FILTER_SIZE] = keyRate;
    lastGpukeyRate[filterPos%FILTER_SIZE] = gpuKeyRate;
    filterPos++;

    // KeyRate smoothing
    double avgKeyRate = 0.0;
    double avgGpuKeyRate = 0.0;
    for (int i = 0; i<FILTER_SIZE; i++) {
      avgKeyRate += lastkeyRate[i];
      avgGpuKeyRate += lastGpukeyRate[i];
    }
    avgKeyRate /= (double)(FILTER_SIZE);
    avgGpuKeyRate /= (double)(FILTER_SIZE);

    if (isAlive(params)) {
      printf("%.3f MK/s (GPU %.3f MK/s) (2^%.2f) %s[%d]\r",
        avgKeyRate / 1000000.0, avgGpuKeyRate / 1000000.0,
          log2((double)count), GetExpectedTime(avgKeyRate, (double)count).c_str(),nbFoundKey);
    }

    lastCount = count;
    lastGPUCount = gpuCount;
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
