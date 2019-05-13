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

#include "Wildcard.h"

using namespace std;


bool Wildcard::match(const char *str, const char *pattern, bool caseSensitive) {

  const char *s;
  const char *p;
  bool star = false;

loopStart:
  for (s = str, p = pattern; *s; ++s, ++p) {

    switch (*p) {
    case '?':
      if (*s == '.') goto starCheck;
      break;

    case '*':
      star = true;
      str = s, pattern = p;
      if (!*++pattern) return true;
      goto loopStart;

    default:
      if (caseSensitive) {
        if (*s != *p)
          goto starCheck;
      } else {
        if (tolower(*s) != tolower(*p))
          goto starCheck;
      }
      break;
    } /* endswitch */

  } /* endfor */

  if (*p == '*') ++p;
  return (!*p);

starCheck:
  if (!star) return false;
  str++;
  goto loopStart;

}
