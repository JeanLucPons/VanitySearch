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

#include "Base58.h"

#include <algorithm>
#include <string.h>
#include <cstdint>

/** All alphanumeric characters except for "0", "I", "O", and "l" */
static const char* pszBase58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

static const int8_t b58digits_map[] = {
	-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
	-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
	-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
	-1, 0, 1, 2, 3, 4, 5, 6,  7, 8,-1,-1,-1,-1,-1,-1,
	-1, 9,10,11,12,13,14,15, 16,-1,17,18,19,20,21,-1,
	22,23,24,25,26,27,28,29, 30,31,32,-1,-1,-1,-1,-1,
	-1,33,34,35,36,37,38,39, 40,41,42,43,-1,44,45,46,
	47,48,49,50,51,52,53,54, 55,56,57,-1,-1,-1,-1,-1,
};

bool DecodeBase58(const char* psz, std::vector<uint8_t> &vch) {

  uint8_t digits[256];

  // Skip and count leading '1'
  int zeroes = 0;
  while (*psz == '1') {
    zeroes++;
    psz++;
  }

  int length = (int)strlen(psz);

  // Process the characters	
  int digitslen = 1;
  digits[0] = 0;
  for (int i = 0; i < length; i++) {

    // Decode base58 character
    if (psz[i] & 0x80)
      return false;

    int8_t c = b58digits_map[psz[i]];
    if (c < 0)
      return false;

    uint32_t carry = (uint32_t)c;
    for (int j = 0; j < digitslen; j++) {
      carry += (uint32_t)(digits[j]) * 58;
      digits[j] = (uint8_t)(carry % 256);
      carry /= 256;
    }
    while (carry > 0) {
      digits[digitslen++] = (uint8_t)(carry % 256);
      carry /= 256;
    }

  }

  vch.clear();
  vch.reserve(zeroes + digitslen);
  // zeros
  for (int i = 0; i < zeroes; i++)
    vch.push_back(0);

  // reverse    
  for (int i = 0; i < digitslen; i++)
    vch.push_back(digits[digitslen - 1 - i]);

  return true;

}

std::string EncodeBase58(const unsigned char* pbegin, const unsigned char* pend) {

  std::string ret;
  unsigned char digits[256];
  
  // Skip leading zeroes
  while (pbegin != pend && *pbegin == 0) {
    ret.push_back('1');
    pbegin++;
  }
  int length = (int)(pend - pbegin);

  int digitslen = 1;
  digits[0]=0;
  for(int i = 0; i < length; i++) {
    uint32_t carry = pbegin[i];
    for(int j = 0; j < digitslen; j++) {
      carry += (uint32_t)(digits[j]) << 8;
      digits[j] = (unsigned char)(carry % 58);
      carry /= 58;
    }
    while(carry > 0) {
      digits[digitslen++] = (unsigned char)(carry % 58);
      carry /= 58;
    }
  }

  // reverse
  for(int i = 0; i < digitslen; i++)
    ret.push_back(pszBase58[digits[digitslen - 1 - i]]);

  return ret;

}

std::string EncodeBase58(const std::vector<unsigned char>& vch)
{
    return EncodeBase58(vch.data(), vch.data() + vch.size());
}

bool DecodeBase58(const std::string& str, std::vector<unsigned char>& vchRet)
{
    return DecodeBase58(str.c_str(), vchRet);
}

