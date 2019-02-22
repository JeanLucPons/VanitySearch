# VanitySearch

VanitySearch is a bitcoin address prefix finder. It uses fixed size arithmethic in order to get best performances. 
Secure hash algorithms (SHA256 and RIPEMD160) are performed using SSE on the CPU. The GPU kernel has been written using
CUDA in order to take advantage of inline PTX assembly. VanitySearch may not compute a good grid size for your hardware, so try different values using -g option. If you want to use GPU and CPU together, you may have best performance by keeping one CPU core for handling GPU/CPU exchanges (use -t option to set the number of CPU threads).
Linux release does not support GPU yet.

# Usage

You can downlad latest release from https://github.com/JeanLucPons/VanitySearch/releases

VanitySeacrh [-check] [-v] [-u] [-gpu] [-stop] [-o outputfile] [-gpuId gpuId] [-g gridSize] [-s seed] [-t threadNumber] prefix prefix: prefix to search
 -v: Print version
 -check: Check GPU kernel vs CPU
 -u: Search uncompressed addresses
 -o outputfile: Output results to the specified file
 -gpu: Enable gpu calculation
 -gpu gpuId: Use gpu gpuId, default is 0
 -g gridSize: Specify GPU kernel gridsize, default is 16*(MP number)
 -s seed: Specify a seed for the base key, default is random
 -t threadNumber: Specify number of CPU thread, default is number of core
 -stop: Stop when prefix is found
  
  Exemple (Windows, Intel Core i7-4770 3.4GHz 8 multithreaded cores, GeForce GTX 645):
  ```
  C:\C++\VanitySearch\x64\Release>VanitySearch.exe -t 6 -gpu 1tryme
  Start Fri Feb 22 10:30:57 2019
  Search: 1tryme
  Difficulty: 15318045009
  Base Key:DD62DD7AD67A8BF2AED0A64605CC54A54EB678A9B4FB4631BDC5966481FF1E01
  Number of CPU thread: 6
  GPU: GPU #0 GeForce GTX 645 (3x192 cores) Grid(48x64)
  24.637 MK/s (GPU 18.480 MK/s) (2^29.40) [P 4.51%][50.00% in 00:06:42][0]
  Pub Addr: 1trymepuUbhT7FYVDZwUQAkvc7Bxyoc8B
  Prv Addr: 5KVnbSfWEAiK5Gpf5uV2SpSNykHQoCo9fLLEne7SAYPUQuj2dMK
  Prv Key : 0xDD62DD7AD67A8BF2AED0A64605CC54A54EB678A9B4FB4632BDC5966483C7A574
  Check   : 1Hn7uyytuQdUC2c54WFP3ZYSvsJ4fUMwfu
  Check   : 1trymepuUbhT7FYVDZwUQAkvc7Bxyoc8B (comp)
  ```

# License

VanitySearch is licensed under GPLv3.

