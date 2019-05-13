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
  static std::string getSeed(int size);
  static void SleepMillis(uint32_t millis);

#ifdef WIN64
  static LARGE_INTEGER perfTickStart;
  static double perfTicksPerSec;
  static LARGE_INTEGER qwTicksPerSec;
#else
  static time_t tickStart;
#endif

};

#endif // TIMERH
