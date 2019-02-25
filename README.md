# VanitySearch

VanitySearch is a bitcoin address prefix finder. It uses fixed size arithmethic in order to get best performances. 
Secure hash algorithms (SHA256 and RIPEMD160) are performed using SSE on the CPU. The GPU kernel has been written using
CUDA in order to take advantage of inline PTX assembly. VanitySearch may not compute a good grid size for your hardware, so try different values using -g option. If you want to use GPU and CPU together, you may have best performance by keeping one CPU core for handling GPU/CPU exchanges (use -t option to set the number of CPU threads).
Linux release does not support GPU yet.

# Usage

You can downlad latest release from https://github.com/JeanLucPons/VanitySearch/releases

VanitySeacrh [-check] [-v] [-u] [-gpu] [-stop] [-o outputfile] [-gpuId gpuId] [-g gridSize] [-s seed] [-t threadNumber] prefix
  prefix: prefix to search\
  -v: Print version\
  -check: Check GPU kernel vs CPU\
  -u: Search uncompressed addresses\
  -o outputfile: Output results to the specified file\
  -gpu: Enable gpu calculation\
  -gpu gpuId: Use gpu gpuId, default is 0\
  -g gridSize: Specify GPU kernel gridsize, default is 16*(MP number)\
  -s seed: Specify a seed for the base key, default is random\
  -t threadNumber: Specify number of CPU thread, default is number of core\
  -stop: Stop when prefix is found
  
  Exemple (Windows, Intel Core i7-4770 3.4GHz 8 multithreaded cores, GeForce GTX 645):
  ```
  C:\C++\VanitySearch\x64\Release>VanitySearch.exe -stop -gpu 1tryme
  Start Mon Feb 25 08:10:36 2019
  Search: 1tryme
  Difficulty: 15318045009
  Base Key:45D090A251EB279D63E3632DA1FDB01A0052A3209ED841A17CD4F2E7C9583D0A
  Number of CPU thread: 7
  GPU: GPU #0 GeForce GTX 645 (3x192 cores) Grid(48x64)
  31.385 MK/s (GPU 24.378 MK/s) (2^35.04) [P 90.00%][99.00% in 00:18:43]
  Pub Addr: 1trymeffDV5eAJMANdFs3vVAq6oEjdzK6
  Prv Addr: 5JM2uQyqRAzc1Jusep4gfKFRurRA49EEKpzByi5ESpp4KT9PBVZ
  Prv Key : 0x45D090A251EB279D63E3632DA1FDB01A0052A3209ED841A77CD4F2E80C2F92D4
  Check   : 16rXQ3Rvzb1oxdTn1JSPbYXBniPU1uzHBe
  Check   : 1trymeffDV5eAJMANdFs3vVAq6oEjdzK6 (comp)
  ```

# License

VanitySearch is licensed under GPLv3.

