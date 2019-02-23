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
  Start Sat Feb 23 08:55:41 2019
  Search: 1tryme
  Difficulty: 15318045009
  Base Key:3E9F14F4D9D9CFC99C83AC9925C500548D1A1F0160060F2B6E6D0F50B8DB86FC
  Number of CPU thread: 7
  GPU: GPU #0 GeForce GTX 645 (3x192 cores) Grid(48x64)
  27.183 MK/s (GPU 20.446 MK/s) (2^32.24) [P 28.25%][50.00% in 00:03:23]
  Pub Addr: 1trymencMnwgP9L3uAvJ5e7XFbMcz4iFR
  Prv Addr: 5JHsAKSYZZ4dyiQWrpiNYDEW4sxkkbCcKJWQAsKnBre2xUbKTTA
  Prv Key : 0x3E9F14F4D9D9CFC99C83AC9925C500548D1A26E260060F2B6E6D0F50B8EE7D03
  Check   : 1P651E3ASWKpJTeM1L6tz1RSaCpVzpUWkq
  Check   : 1trymencMnwgP9L3uAvJ5e7XFbMcz4iFR (comp)
  ```

# License

VanitySearch is licensed under GPLv3.

