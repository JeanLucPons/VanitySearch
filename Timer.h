#ifndef TIMERH
#define TIMERH

#include <time.h>
#include <string>
#ifdef WIN64
#include <windows.h>
#endif

class Timer {

public:
  static void Init();
  static double get_tick();
  static void printResult(char *unit, int nbTry, double t0, double t1);
  static std::string getResult(char *unit, int nbTry, double t0, double t1);
  static int getCoreNumber();

#ifdef WIN64
  static LARGE_INTEGER perfTickStart;
  static double perfTicksPerSec;
  static LARGE_INTEGER qwTicksPerSec;
#else
  static time_t tickStart;
#endif

};

#endif // TIMERH
