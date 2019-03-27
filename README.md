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

# Discussion Thread

[Disucussion about VanitySearch@bitcointalk](https://bitcointalk.org/index.php?topic=5112311.0)

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
C:\C++\VanitySearch\x64\Release>VanitySearch.exe -stop -gpu 1TryMe
VanitySearch v1.10
Difficulty: 15318045009
Search: 1TryMe [Compressed]
Start Wed Mar 27 10:19:04 2019
Base Key:681E8A2A2F580C427EF025A2CFE556E07EA7055B2A92B6234DB0B5AAFDA2E5FF
Number of CPU thread: 7
GPU: GPU #0 GeForce GTX 645 (3x192 cores) Grid(24x128)
59.567 MK/s (GPU 43.544 MK/s) (2^33.71) [P 59.98%][60.00% in 00:00:00][0]
Pub Addr: 1TryMeGTxspHHGDWRX5KD8zPPM6KoNdfV
Priv (WIF): L4hqNRo8dwoWN9GWCSuFagrbsysUpbLxG3rDXJRCXGubC3oPB62z
Priv (HEX): 0xDF546BDE70567A14A83AC69B17EF5845239F39B35820243DC3859AC46894DCDF
```

# Trying to attack a list of addresses

The bitcoin address (P2PKH) consists of a hash160 (displayed in Base58 format) which means that there are 2<sup>160</sup> possible addresses. A secure hash function can be seen as a pseudo number generator, it transforms a given message in a random number. In this case, a number (uniformaly distributed) in the range [0,2<sup>160</sup>]. So, the probability to hit a particular number after n tries is 1-(1-1/2<sup>160</sup>)<sup>n</sup>. We perform n Bernoulli trials statistically independent.\
If we have a list of m distinct addresses (m<=2<sup>160</sup>), the search space is then reduced to 2<sup>160</sup>/m, the probability to find a collision after 1 try becomes m/2<sup>160</sup> and the probability to find a collision after n tries becomes 1-(1-m/2<sup>160</sup>)<sup>n</sup>.\
An example:\
We have a hardware capable of generating **1GKey/s** and we have an input list of **10<sup>6</sup>** addresses, the following table shows the probability of finding a collision after a certain amount of time:
  
| Time     |  Probability  |
|----------|:-------------:|
| 1 second |6.8e-34|
| 1 minute |4e-32|
| 1 hour |2.4e-30|
| 1 day |5.9e-29|
| 1 year |2.1e-26|
| 10 years | 2.1e-25 |
| 1000 years | 2.1e-23 |
| Age of earth | 8.64e-17 |
| Age of universe | 2.8e-16 (much less than winning at the lottery) |

Calculation has been done using this [online high precision calculator](https://keisan.casio.com/calculator)

As you can see, even with a competitive hardware, it is very unlikely that you find a collision. Birthday paradox doesn't apply in this context, it works only if we know already the public key (not the address, the hash of the public key) we want to find.  This program doesn't look for collisions between public keys. It searchs only for collisions with addresses with a certain prefix. 

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
Depenging on the CUDA SDK version and on your Linux distribution you may need to install an older g++ (just for the CUDA SDK).\
Edit the makefile and set up the good CUDA SDK path and appropriate compiler for nvcc. 

```
CUDA       = /usr/local/cuda-8.0
CXXCUDA    = /usr/bin/g++-4.8
```

You can enter a list of architectrure (refer to nvcc documentation) if you have several GPU with different architecture. Compute capability 2.0 (Fermi) is deprecated for recent CUDA SDK.
VanitySearch need to be compiled and linked with a recent gcc (>=7). The current release has been compiled with gcc 7.3.0.\
Go to the VanitySearch directory. ccap is the desired compute capability.

```
$ g++ -v
gcc version 7.3.0 (Ubuntu 7.3.0-27ubuntu1~18.04)
$ make all (for build without CUDA support)
or
$ make gpu=1 ccap=20 all
```
Runnig VanitySearch (Intel(R) Xeon(R) CPU, 8 cores,  @ 2.93GHz, Quadro 600 (x2))
```
$export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
pons@linpons:~/VanitySearch$ ./VanitySearch -t 7 -gpu -gpuId 0,1 1TryMe
VanitySearch v1.10
Difficulty: 15318045009
Search: 1TryMe [Compressed]
Start Wed Mar 27 10:26:43 2019
Base Key:C6718D8E50C1A5877DE3E52021C116F7598826873C61496BDB7CAD668CE3DCE5
Number of CPU thread: 7
GPU: GPU #1 Quadro 600 (2x48 cores) Grid(16x128)
GPU: GPU #0 Quadro 600 (2x48 cores) Grid(16x128)
40.284 MK/s (GPU 27.520 MK/s) (2^31.84) [P 22.24%][50.00% in 00:02:47][0]  
Pub Addr: 1TryMeERTZK7RCTemSJB5SNb2WcKSx45p
Priv (WIF): Ky9bMLDpb9o5rBwHtLaidREyA6NzLFkWJ19QjPDe2XDYJdmdUsRk
Priv (HEX): 0x398E7271AF3E5A78821C1ADFDE3EE90760A6B65F72D856CFE455B1264350BCE8
```

# License

VanitySearch is licensed under GPLv3.

