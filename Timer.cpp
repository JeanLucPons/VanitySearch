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

static const char *prefix[] = { "","Kilo","Mega","Giga","Tera","Peta","Hexa" };

#ifdef WIN64

LARGE_INTEGER Timer::perfTickStart;
double Timer::perfTicksPerSec;
LARGE_INTEGER Timer::qwTicksPerSec;
#include <wincrypt.h>

#else

#include <sys/time.h>
#include <unistd.h>
#include <string.h>
time_t Timer::tickStart;

#endif

void Timer::Init() {

#ifdef WIN64
  QueryPerformanceFrequency(&qwTicksPerSec);
  QueryPerformanceCounter(&perfTickStart);
  perfTicksPerSec = (double)qwTicksPerSec.QuadPart;
#else
  tickStart=time(NULL);
#endif

}

double Timer::get_tick() {

#ifdef WIN64
  LARGE_INTEGER t, dt;
  QueryPerformanceCounter(&t);
  dt.QuadPart = t.QuadPart - perfTickStart.QuadPart;
  return (double)(dt.QuadPart) / perfTicksPerSec;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)(tv.tv_sec - tickStart) + (double)tv.tv_usec / 1e6;
#endif

}

std::string Timer::getSeed(int size) {

  std::string ret;
  char tmp[3];
  unsigned char *buff = (unsigned char *)malloc(size);

#ifdef WIN64

  HCRYPTPROV   hCryptProv = NULL;
  LPCSTR UserName = "KeyContainer";

  if (!CryptAcquireContext(
    &hCryptProv,               // handle to the CSP
    UserName,                  // container name
    NULL,                      // use the default provider
    PROV_RSA_FULL,             // provider type
    0))                        // flag values
  {
    //-------------------------------------------------------------------
    // An error occurred in acquiring the context. This could mean
    // that the key container requested does not exist. In this case,
    // the function can be called again to attempt to create a new key
    // container. Error codes are defined in Winerror.h.
    if (GetLastError() == NTE_BAD_KEYSET) {
      if (!CryptAcquireContext(
        &hCryptProv,
        UserName,
        NULL,
        PROV_RSA_FULL,
        CRYPT_NEWKEYSET)) {
        printf("CryptAcquireContext(): Could not create a new key container.\n");
        exit(1);
      }
    } else {
      printf("CryptAcquireContext(): A cryptographic service handle could not be acquired.\n");
      exit(1);
    }
  }

  if (!CryptGenRandom(hCryptProv,size,buff)) {
    printf("CryptGenRandom(): Error during random sequence acquisition.\n");
    exit(1);
  }

  CryptReleaseContext(hCryptProv, 0);

#else

  FILE *f = fopen("/dev/urandom","rb");
  if(f==NULL) {
    printf("Failed to open /dev/urandom %s\n", strerror( errno ));
    exit(1);
  }
  if( fread(buff,1,size,f)!=size ) {
    printf("Failed to read from /dev/urandom %s\n", strerror( errno ));
    exit(1);
  }
  fclose(f);

#endif

  for (int i = 0; i < size; i++) {
    sprintf(tmp,"%02X",buff[i]);
    ret.append(tmp);
  }

  free(buff);
  return ret;

}

uint32_t Timer::getSeed32() {
  return ::strtoul(getSeed(4).c_str(),NULL,16);
}

std::string Timer::getResult(char *unit, int nbTry, double t0, double t1) {

  char tmp[256];
  int pIdx = 0;
  double nbCallPerSec = (double)nbTry / (t1 - t0);
  while (nbCallPerSec > 1000.0 && pIdx < 5) {
    pIdx++;
    nbCallPerSec = nbCallPerSec / 1000.0;
  }
  sprintf(tmp, "%.3f %s%s/sec", nbCallPerSec, prefix[pIdx], unit);
  return std::string(tmp);

}

void Timer::printResult(char *unit, int nbTry, double t0, double t1) {

  printf("%s\n", getResult(unit, nbTry, t0, t1).c_str());

}

int Timer::getCoreNumber() {

#ifdef WIN64
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  return sysinfo.dwNumberOfProcessors;
#else
  // TODO
  return 1;
#endif

}

void Timer::SleepMillis(uint32_t millis) {

#ifdef WIN64
  Sleep(millis);
#else
  usleep(millis*1000);
#endif

}
