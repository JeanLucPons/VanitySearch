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

// CUDA Kernel main function
// Compute SecpK1 keys and calculate RIPEMD160(SHA256(key)) then check prefix
// For the kernel, we use a 16 bits prefix which correspond to ~3 Base58 characters
// Using 32bit prefix for long prefix search does not bring significant performance improvements
// (The CPU computes the full address and check the full prefix)
// 
// We use affine coordinates for elliptic curve point (ie Z=1)



// Probabilty to lost a prefix can be calculated using Psk function
// Even if a 16 bits prefix is lost, it is unlikely that this prefix matches the desired address
// unless the desired prefix is short but in that case, an other prefix will be found quickly
#define CHECK_PREFIX(incr)   \
__HASHFUNC__(px, py, hash);  \
pr0 = *(prefix_t *)hash;     \
if (pr0 == sPrefix) {        \
  if (nbFound < MAX_FOUND) { \
    uint16_t _icr = (incr);   \
    memcpy(out + (1 + nbFound * ITEM_SIZE), &_icr, 2);     \
    memcpy(out + (1 + nbFound * ITEM_SIZE + 2), hash, 20); \
    nbFound++; \
  } \
}

__device__ void __COMPFUNC__(uint64_t *startx, uint64_t *starty, prefix_t sPrefix, uint8_t *out) {

  uint64_t dx[GRP_SIZE/2+1][4];
  uint64_t px[4];
  uint64_t py[4];
  uint64_t sx[4];
  uint64_t sy[4];
  uint64_t dy[4];
  uint64_t _s[4];
  uint64_t _p2[4];
  uint8_t  hash[20];
  uint8_t  nbFound = 0;
  prefix_t pr0;


  for (uint32_t j = 0; j < STEP_SIZE / GRP_SIZE; j++) {

    // Load starting key
    __syncthreads();
    Load256A(sx, startx);
    Load256A(sy, starty);

    // Fill group with delta x
    uint32_t i;
    for (i = 0; i < HSIZE; i++)
      ModSub256(dx[i], Gx[i], sx);
    ModSub256(dx[i] , Gx[i], sx);  // For the first point
    ModSub256(dx[i+1],_2Gnx, sx);  // For the next center point

    // Compute modular inverse
    _ModInvGrouped(dx);

    // We use the fact that P + i*G and P - i*G has the same deltax, so the same inverse
    // We compute key in the positive and negative way from the center of the group

    // Check starting point
    Load256(px, sx);
    Load256(py, sy);
    CHECK_PREFIX(GRP_SIZE / 2);

    for (uint32_t i = 0; i < HSIZE; i++) {

      // P = StartPoint + i*G
      Load256(px, sx);
      Load256(py, sy);
      ModSub256(dy, Gy[i], py);

      _ModMult(_s, dy, dx[i]);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
      _ModMult(_p2, _s, _s);        // _p = pow2(s)

      ModSub256(px, _p2,px);
      ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;

      ModSub256(py, Gx[i], px);
      _ModMult(py, _s);             // py = - s*(ret.x-p2.x)
      ModSub256(py, Gy[i]);         // py = - p2.y - s*(ret.x-p2.x);  

      CHECK_PREFIX(GRP_SIZE / 2 + (i + 1));

      // P = StartPoint - i*G, if (x,y) = i*G then (x,-y) = -i*G
      Load256(px, sx);
      Load256(py, sy);
      ModNeg256(dy,Gy[i]);
      ModSub256(dy, py);

      _ModMult(_s, dy, dx[i]);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
      _ModMult(_p2, _s, _s);        // _p = pow2(s)

      ModSub256(px, _p2, px);
      ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;

      ModSub256(py, Gx[i], px);
      _ModMult(py, _s);             // py = - s*(ret.x-p2.x)
      ModAdd256(py, Gy[i]);         // py = - p2.y - s*(ret.x-p2.x);  

      CHECK_PREFIX(GRP_SIZE / 2 - (i + 1));

    }

    // First point (startP - (GRP_SZIE/2)*G)
    Load256(px, sx);
    Load256(py, sy);
    ModNeg256(dy, Gy[i]);
    ModSub256(dy, py);

    _ModMult(_s, dy, dx[i]);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
    _ModMult(_p2, _s, _s);        // _p = pow2(s)

    ModSub256(px, _p2, px);
    ModSub256(px, Gx[i]);         // px = pow2(s) - p1.x - p2.x;

    ModSub256(py, Gx[i], px);
    _ModMult(py, _s);             // py = - s*(ret.x-p2.x)
    ModAdd256(py, Gy[i]);         // py = - p2.y - s*(ret.x-p2.x);  

    CHECK_PREFIX(0);

    i++;

    // Next start point (startP + GRP_SIZE*G)
    Load256(px, sx);
    Load256(py, sy);
    ModSub256(dy, _2Gny, py);

    _ModMult(_s, dy, dx[i]);      //  s = (p2.y-p1.y)*inverse(p2.x-p1.x)
    _ModMult(_p2, _s, _s);        // _p = pow2(s)

    ModSub256(px, _p2, px);
    ModSub256(px, _2Gnx);         // px = pow2(s) - p1.x - p2.x;

    ModSub256(py, _2Gnx, px);
    _ModMult(py, _s);             // py = - s*(ret.x-p2.x)
    ModSub256(py, _2Gny);         // py = - p2.y - s*(ret.x-p2.x);  

    // Update starting point
    __syncthreads();
    Store256A(startx, px);
    Store256A(starty, py);

  }

  // First index is the number of prefix found
  out[0] = nbFound;

}
