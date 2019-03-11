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
// For the kernel, we use a 16 bits prefix lookup table which correspond to ~3 Base58 characters
// A second level lookup table contains 32 bits prefix (if used)
// (The CPU computes the full address and check the full prefix)
// 
// We use affine coordinates for elliptic curve point (ie Z=1)

#define LOOKUP32(_h)                    \
if (lookup32) {                         \
  off = lookup32[pr0];                  \
  l32 = _h[0];                          \
  found = false;                        \
  i = 0;                                \
  while (!found && (i < hit) &&         \
     (l32 >= lookup32[off + i])) {      \
    found = (lookup32[off + i] == l32); \
    i++;                                \
  }                                     \
  if (!found) return;                   \
}

__device__ __noinline__ void CheckHash(uint32_t mode,prefix_t *prefix,uint64_t *px,uint64_t *py,
                          int32_t incr,uint32_t tid,uint32_t *lookup32,uint32_t *out) {

  uint32_t   h[20];
  uint32_t   hn[20];
  bool       found;
  uint32_t   off;
  prefixl_t  l32;
  prefix_t   pr0;
  prefix_t   hit;
  uint32_t   pos;
  int i;

  if (mode) {
    _GetHash160Comp(px, py, (uint8_t *)h);
    ModNeg256(py);
    _GetHash160Comp(px, py, (uint8_t *)hn);
  } else {
    _GetHash160(px, py, (uint8_t *)h);
    ModNeg256(py);
    _GetHash160(px, py, (uint8_t *)hn);
  }

  // Point
  pr0 = *(prefix_t *)h;
  hit = prefix[pr0];
  if (hit) {

    LOOKUP32(h);
    pos = atomicAdd(out, 1);
    if (pos < MAX_FOUND) {
      out[pos*ITEM_SIZE32 + 1] = tid;
      out[pos*ITEM_SIZE32 + 2] = incr;
      out[pos*ITEM_SIZE32 + 3] = h[0];
      out[pos*ITEM_SIZE32 + 4] = h[1];
      out[pos*ITEM_SIZE32 + 5] = h[2];
      out[pos*ITEM_SIZE32 + 6] = h[3];
      out[pos*ITEM_SIZE32 + 7] = h[4];
    }

  }

  // Symetric Point
  pr0 = *(prefix_t *)hn;
  hit = prefix[pr0];
  if (hit) {

    LOOKUP32(hn);
    pos = atomicAdd(out, 1);
    if (pos < MAX_FOUND) {
      out[pos*ITEM_SIZE32 + 1] = tid;
      out[pos*ITEM_SIZE32 + 2] = -incr;
      out[pos*ITEM_SIZE32 + 3] = hn[0];
      out[pos*ITEM_SIZE32 + 4] = hn[1];
      out[pos*ITEM_SIZE32 + 5] = hn[2];
      out[pos*ITEM_SIZE32 + 6] = hn[3];
      out[pos*ITEM_SIZE32 + 7] = hn[4];
    }

  }

}

#define CHECK_PREFIX(incr) CheckHash(mode, sPrefix, px, py, j*GRP_SIZE + (incr), tid, lookup32, out);

__device__ void ComputeKeys(uint32_t mode, uint64_t *startx, uint64_t *starty, 
                             prefix_t *sPrefix, uint32_t *lookup32, uint32_t *out) {

  uint64_t dx[GRP_SIZE/2+1][4];
  uint64_t px[4];
  uint64_t py[4];
  uint64_t sx[4];
  uint64_t sy[4];
  uint64_t dy[4];
  uint64_t _s[4];
  uint64_t _p2[4];
  uint32_t tid = (blockIdx.x*blockDim.x) + threadIdx.x;

  // Load starting key
  __syncthreads();
  Load256A(sx, startx);
  Load256A(sy, starty);

  for (uint32_t j = 0; j < STEP_SIZE / GRP_SIZE; j++) {

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

    Load256(sx, px);
    Load256(sy, py);

  }

  // Update starting point
  __syncthreads();
  Store256A(startx, sx);
  Store256A(starty, sy);

}
