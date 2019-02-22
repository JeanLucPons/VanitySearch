#include "Timer.h"

static const char *prefix[] = { "","Kilo","Mega","Giga","Tera","Peta","Hexa" };

#ifdef WIN64
LARGE_INTEGER Timer::perfTickStart;
double Timer::perfTicksPerSec;
LARGE_INTEGER Timer::qwTicksPerSec;
#else
#include <sys/time.h>
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

