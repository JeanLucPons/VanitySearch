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
  VanitySeacrh [-check] [-v] [-u] [-gpu] [-stop] [-i inputfile] [-o outputfile] [-gpuId gpuId1[,gpuId2,...]] [-g gridSize1[,gridSize2,...]] [-s seed] [-t threadNumber] prefix
  prefix: prefix to search
  -v: Print version
  -check: Check CPU and GPU kernel vs CPU
  -u: Search uncompressed addresses
  -o outputfile: Output results to the specified file
  -i inputfile: Get list of prefixes to search from specified file
  -gpu: Enable gpu calculation
  -gpu gpuId1,gpuId2,...: List of GPU(s) to use, default is 0
  -g gridSize1,gridSize2,...: Specify GPU(s) kernel gridsize, default is 8*(MP number)
  -s seed: Specify a seed for the base key, default is random
  -t threadNumber: Specify number of CPU thread, default is number of core
  -nosse : Disable SSE hash function
  -l : List cuda enabled devices
  -stop: Stop when prefix is found
  ```
 
  Exemple (Windows, Intel Core i7-4770 3.4GHz 8 multithreaded cores, GeForce GTX 645):
  ```
  C:\C++\VanitySearch\x64\Release>VanitySearch.exe -stop -gpu 1Happy
  Start Mon Mar 11 09:26:22 2019
  Difficulty: 264104224Search: 1Happy
  Base Key:94AB82403B15201B402060E35462957735A583BE7BCBBF360F30BAE1766DA35D
  Number of CPU thread: 7
  GPU: GPU #0 GeForce GTX 645 (3x192 cores) Grid(24x128)
  48.330 MK/s (GPU 34.602 MK/s) (2^28.77) [P 82.35%][90.00% in 00:00:03][0]
  Pub Addr: 1HappycX2ah61mkmkXqMbrB2W85Did1QXU
  Prv Addr: 5JwmA143tk3Sy48HqLWERXQHfkbJt6v2CJ8QEega6dpGhgFw97N
  Prv Key : 0x94AB82403B15201B402060E35462957735A583BE7BCBBF370F30BAE177083B15
  Check   : 1GxrqHh1EnKrdjZxGqGUT1fJMV8RcBPoJW
  Check   : 1HappycX2ah61mkmkXqMbrB2W85Did1QXU (comp)
  ```

# Trying to attack a list of addresses

Please don't use VanitySearch to attack a list of complete addresses. It is very unlikely that you find a collision. The time displayed indicates the time needed to reach the displayed probability of the most proabable prefix in the list. In case of having n complete addresses in the input file, simply divide this time by the number of entries to get an approximative idea of the time needed to reach the displayed probability (in fact it is longer). Even with a file containing 1 billion of addresses, using a very competitive hardware, the time needed to reach a probability of 50% will be much longer than the age of the universe. Note that the birtday paradox cannot be applied here as we look for fixed addresses and there is no trick possible (as for Pollard rho method on points coordinates) to simulate random walks because addresses are hashed.

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
pons@linpons:~/VanitySearch$ ./VanitySearch -stop -gpu -t 7 -gpuId 0,1 1Happy
Start Mon Mar 11 09:15:44 2019
Difficulty: 264104224Search: 1Happy
Base Key:971888F286EA549956BF03C42223D15D96661379399EA9D65831CEDEA6918880
Number of CPU thread: 7
GPU: GPU #1 Quadro 600 (2x48 cores) Grid(16x128)
GPU: GPU #0 Quadro 600 (2x48 cores) Grid(16x128)
31.315 MK/s (GPU 19.921 MK/s) (2^28.11) [P 66.50%][70.00% in 00:00:00][0]
Pub Addr: 1Happysu9MV2H8BbjFT9MRS1jbEqw1Qs3P
Prv Addr: 5JcVDZtPPU1PF1drPyHR5uf8VJvMf7qZE1VmPfSRA8ycdhNx4RL
Prv Key : 0x68E7770D7915AB66A940FC3BDDDC2EA123C7C96D71D0F66567A08FAE29A4680E
Check   : 1ECSMNQNtejgPqEBDp3rRUx1mZuiyQ7QET
Check   : 1Happysu9MV2H8BbjFT9MRS1jbEqw1Qs3P (comp)
```

# License

VanitySearch is licensed under GPLv3.

