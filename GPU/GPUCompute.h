// CUDA Kernel main function
// Compute SecpK1 keys and calculate RIPEMD160(SHA256(key)) then check prefix
// For the kernel, we use a 16 bits prefix which correspond to ~3 Base58 characters
// Using 32bit prefix for long prefix search does not bring significant performance improvements
// (The CPU computes the full address and check the full prefix)
// 
// We use affine coordinates for elliptic curve point (ie Z=1)

__device__ void __COMPFUNC__(uint64_t *startx, uint64_t *starty, prefix_t sPrefix, uint8_t *out) {

  uint64_t dx[GRP_SIZE][4];
  uint64_t px[4];
  uint64_t py[4];
  uint64_t dy[4];
  uint64_t _s[4];
  uint64_t _p2[4];
  uint8_t  hash[20];
  uint8_t  nbFound = 0;

  // Load starting key
  Load256A(px, startx);
  Load256A(py, starty);

  for (uint32_t j = 0; j < STEP_SIZE / GRP_SIZE; j++) {

    // Fill group with delta x
    for (uint32_t i = 0; i < GRP_SIZE; i++)
      ModSub256(dx[i], Gx[i], px);

    // Compute modular inverse
    _ModInvGrouped(dx);

    for (uint32_t i = 0; i < GRP_SIZE; i++) {

      __HASHFUNC__(px, py, hash);
      prefix_t pr0 = *(prefix_t *)hash;
      if (pr0 == sPrefix) {
        // We found a matching prefix
        // Probabilty to lost a prefix can be calculated using Psk function
        // Even if a 16 bits prefix is lost, it is unlikely that this prefix matches the desired address
        // unless the desired prefix is short but in that case, an other prefix will be found quickly
        if (nbFound < MAX_FOUND) {
          uint16_t incr = j * GRP_SIZE + i;
          memcpy(out + (1 + nbFound * ITEM_SIZE), &incr, 2);
          memcpy(out + (1 + nbFound * ITEM_SIZE + 2), hash, 20);
          nbFound++;
        }
      }

      __syncthreads();
      Load256A(px, startx);
      Load256A(py, starty);

      // Compute next point on the curve (P = StartPoint + i*G)
      ModSub256(dy, Gy[i], py);

      _MontgomeryMult(_s, dy, dx[i]);      // s = (p2.y-p1.y)*inverse(p2.x-p1.x);
      _MontgomeryMult(_p2, _s, _s);        // _p = pow2(s)*R^-3
      _MontgomeryMult(_p2, _R4);           // _p = pow2(s)

      ModSub256(px, _p2,px);
      ModSub256(px, Gx[i]);                // px = pow2(s) - p1.x - p2.x;

      ModSub256(py, Gx[i], px);
      _MontgomeryMult(py, _s);
      _MontgomeryMult(py, _R3);
      ModSub256(py, Gy[i]);               // py = - p2.y - s*(ret.x-p2.x);  

    }

    // Update starting point
    __syncthreads();
    Store256A(startx, px);
    Store256A(starty, py);

  }

  // First index is the number of prefix found
  out[0] = nbFound;

}
