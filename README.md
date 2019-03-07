# VanitySearch

VanitySearch is a bitcoin address prefix finder. If you want to generate safe private keys, use the -s option to enter your passphrase which will be used for generating a base key as for BIP38 standard (*VanitySeacrh.exe -s "My PassPhrase" 1MyPrefix*).\
VanitySearch may not compute a good grid size for your GPU, so try different values using -g option in order to get the best performances. If you want to use GPUs and CPUs together, you may have best performances by keeping one CPU core for handling GPU(s)/CPU exchanges (use -t option to set the number of CPU threads).

# Feature

<ul>
  <li>Fixed size arithmetic</li>
  <li>Fast Modular Inversion (Delayed Right Shift 62 bits)</li>
  <li>SecpK1 Fast modular multiplication (2 steps folding 512bits to 256bits using 64 bits digits)</li>
  <li>Use some properties of elliptic curve to generate more keys</li>
  <li>SSE Secure Hash Algorithm SHA256 and RIPEMD160 (CPU)</li>
  <li>Multi-GPU support</li>
  <li>CUDA optimisation via inline PTX assembly</li>
  <li>Seed protected by pbkdf2_hmac_sha512 (BIP38)</li>
</ul>

# Usage

You can downlad latest release from https://github.com/JeanLucPons/VanitySearch/releases

  ```
  VanitySeacrh [-check] [-v] [-u] [-gpu] [-stop] [-o outputfile] [-gpuId gpuId] [-g gridSize] [-s seed] [-t threadNumber] prefix
 prefix: prefix to search
 -v: Print version
 -check: Check CPU and GPU kernel vs CPU
 -u: Search uncompressed addresses
 -o outputfile: Output results to the specified file
 -gpu: Enable gpu calculation
 -gpu gpuId1,gpuId2,...: List of GPU(s) to use, default is 0
 -g gridSize1,gridSize2,...: Specify GPU(s) kernel gridsize, default is 16*(MP number)
 -s seed: Specify a seed for the base key, default is random
 -t threadNumber: Specify number of CPU thread, default is number of core
 -nosse : Disable SSE hash function
 -l : List cuda enabled devices
 -stop: Stop when prefix is found
  ```
 
  Exemple (Windows, Intel Core i7-4770 3.4GHz 8 multithreaded cores, GeForce GTX 645):
  ```
  C:\C++\VanitySearch\x64\Release>VanitySearch.exe -stop -gpu 1Happy
  Start Wed Mar  6 15:29:00 2019
  Search: 1Happy
  Difficulty: 264104224
  Base Key:FED6C568C2E57730BF38D07FD4489C2BE095D2861C00A653492621D2434306B9
  Number of CPU thread: 7
  GPU: GPU #0 GeForce GTX 645 (3x192 cores) Grid(24x128)
  41.641 MK/s (GPU 30.670 MK/s) (2^27.76) [P 57.71%][60.00% in 00:00:00]
  Pub Addr: 1HappydNsxC6mueAXMHG6AzBQJWyBaF7QN
  Prv Addr: 5KkX6nAeQr8efUWoom3AeSLBGj3wJC6bceeuxiSJM7psXjwbJFh
  Prv Key : 0xFED6C568C2E57730BF38D07FD4489C2BE115D286237FA653492621D24343D7CA
  Check   : 13QQPP6PaRnDLpQPhAy2YwfWMy8MZ473q9
  Check   : 1HappydNsxC6mueAXMHG6AzBQJWyBaF7QN (comp)
  ```

# Compilation

## Windows

Intall CUDA SDK and open VanitySearch.sln in Visual C++ 2017.\
You may need to reset your *Windows SDK version* in project properties.\
In Build->Configuration Manager, select the *Release* configuration.\
Build and enjoy.\
\
Note: The current relase has been compiled with CUDA SDK 10.0, if you have a different release of the CUDA SDK, you may need to update CUDA SDK paths in VanitySearch.vcxproj using a text editor. The current nvcc option are set up to architecture starting at 3.0 capability, for older hardware, add the desired compute capabilities to the list in GPUEngine.cu properties, CUDA C/C++, Device, Code Generation.

## Linux

Intall CUDA SDK.\
Depenging on the CUDA SDK version and on your Linux distribution you may need to install an older gcc (just for the CUDA SDK).\
Add a link to the good gcc in /usr/local/cuda, nvcc will use this path first.

```
lrwxrwxrwx 1 root root      16 mars   1 10:54 /usr/local/cuda/bin/g++ -> /usr/bin/g++-4.8*
lrwxrwxrwx 1 root root      16 mars   1 10:53 /usr/local/cuda/bin/gcc -> /usr/bin/gcc-4.8*
```

Edit the makefile and set up the good compute capabilites for your hardware and CUDA SDK path. You can enter a list of architectrure (refer to nvcc documentation). Here it is set up for compute capability 2.0 (Fermi) which is deprecated for recent CUDA SDK.
```
-gencode=arch=compute_20,code=sm_20
```

VanitySearch need to be compiled and linked with a recent gcc. The current release has been compiled with gcc 7.3.0.\
Go to the VanitySearch directory.

```
$ g++ -v
gcc version 7.3.0 (Ubuntu 7.3.0-27ubuntu1~18.04)
$ make all (for build without CUDA support)
or
$ make gpu=1 all
```
Runnig VanitySearch (Intel(R) Xeon(R) CPU, 8 cores,  @ 2.93GHz, Quadro 600 (x2))
```
$export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
pons@linpons:~/VanitySearch$ ./VanitySearch -stop -t 7 -gpu -gpuId 0,1 1Happy
Start Wed Mar  6 15:26:23 2019
Search: 1Happy
Difficulty: 264104224
Base Key:3840A1BDE4792771D67CBFBCB75F2EF08644CDCB93A20A2F953EA289D5785F42
Number of CPU thread: 7
GPU: GPU #1 Quadro 600 (2x48 cores) Grid(16x128)
GPU: GPU #0 Quadro 600 (2x48 cores) Grid(16x128)
24.621 MK/s (GPU 15.726 MK/s) (2^26.99) [P 39.60%][50.00% in 00:00:02]
Pub Addr: 1HappyE4YFXy9NKv47wLSNNxhQ8pHmmnS4
Prv Addr: 5JF4UXBP9PF68hZGHidd39sWBpVfvvGuUaMJXq5b68JAh97FXiS
Prv Key : 0x3840A1BDE4792771D67CBFBCB75F2EF08644CDCB93A20A33953EA289D6109E5C
Check   : 1CP8uGjwfmHfNuXRbSCfw74445CizNjXcA
Check   : 1HappyE4YFXy9NKv47wLSNNxhQ8pHmmnS4 (comp)
```

# License

VanitySearch is licensed under GPLv3.

