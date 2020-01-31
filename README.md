# VanitySearch

VanitySearch is a bitcoin address prefix finder. If you want to generate safe private keys, use the -s option to enter your passphrase which will be used for generating a base key as for BIP38 standard (*VanitySeacrh.exe -s "My PassPhrase" 1MyPrefix*). You can also use *VanitySeacrh.exe -ps "My PassPhrase"* which will add a crypto secure seed to your passphrase.\
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
  <li>Support P2PKH, P2SH and BECH32 addresses</li>
  <li>Support split-key vanity address</li>    
</ul>

# Discussion Thread

[Disucussion about VanitySearch@bitcointalk](https://bitcointalk.org/index.php?topic=5112311.0)

# Usage

You can downlad latest release from https://github.com/JeanLucPons/VanitySearch/releases

```
VanitySeacrh [-check] [-v] [-u] [-b] [-c] [-gpu] [-stop] [-i inputfile]
             [-gpuId gpuId1[,gpuId2,...]] [-g g1x,g1y,[,g2x,g2y,...]]
             [-o outputfile] [-m maxFound] [-ps seed] [-s seed] [-t nbThread]
             [-nosse] [-r rekey] [-check] [-kp] [-sp startPubKey]
             [-rp privkey partialkeyfile] [prefix]

 prefix: prefix to search (Can contains wildcard '?' or '*')
 -v: Print version
 -u: Search uncompressed addresses
 -b: Search both uncompressed or compressed addresses
 -c: Case unsensitive search
 -gpu: Enable gpu calculation
 -stop: Stop when all prefixes are found
 -i inputfile: Get list of prefixes to search from specified file
 -o outputfile: Output results to the specified file
 -gpu gpuId1,gpuId2,...: List of GPU(s) to use, default is 0
 -g g1x,g1y,g2x,g2y, ...: Specify GPU(s) kernel gridsize, default is 8*(MP number),128
 -m: Specify maximun number of prefixes found by each kernel call
 -s seed: Specify a seed for the base key, default is random
 -ps seed: Specify a seed concatened with a crypto secure random seed
 -t threadNumber: Specify number of CPU thread, default is number of core
 -nosse: Disable SSE hash function
 -l: List cuda enabled devices
 -check: Check CPU and GPU kernel vs CPU
 -cp privKey: Compute public key (privKey in hex hormat)
 -kp: Generate key pair
 -rp privkey partialkeyfile: Reconstruct final private key(s) from partial key(s) info.
 -sp startPubKey: Start the search with a pubKey (for private key splitting)
 -r rekey: Rekey interval in MegaKey, default is disabled
```
 
Exemple (Windows, Intel Core i7-4770 3.4GHz 8 multithreaded cores, GeForce GTX 1050 Ti):

```
C:\C++\VanitySearch\x64\Release>VanitySearch.exe -stop -gpu 1TryMe
VanitySearch v1.17
Difficulty: 15318045009
Search: 1TryMe [Compressed]
Start Fri Jan 31 08:12:19 2020
Base Key: DA12E013325F12D6B68520E327847218128B788E6A9F2247BC104A0EE2818F44
Number of CPU thread: 7
GPU: GPU #0 GeForce GTX 1050 Ti (6x128 cores) Grid(48x128)
[251.82 Mkey/s][GPU 235.91 Mkey/s][Total 2^32.82][Prob 39.1%][50% in 00:00:12][Found 0]
PubAddress: 1TryMeJT7cfs4M6csEyhWVQJPAPmJ4NGw
Priv (WIF): p2pkh:Kxs4iWcqYHGBfzVpH4K94STNMHHz72DjaCuNdZeM5VMiP9zxMg15
Priv (HEX): 0x310DBFD6AAB6A63FC71CAB1150A0305ECABBE46819641D2594155CD41D081AF1
```

```
C:\C++\VanitySearch\x64\Release>VanitySearch.exe -stop -gpu 3MyCoin
VanitySearch v1.11
Difficulty: 15318045009
Search: 3MyCoin [Compressed]
Start Wed Apr  3 14:52:45 2019
Base Key:FAF4F856077398AE087372110BF47A1A713C8F94B19CDD962D240B6A853CAD8B
Number of CPU thread: 7
GPU: GPU #0 GeForce GTX 1050 Ti (6x128 cores) Grid(48x128)
124.232 MK/s (GPU 115.601 MK/s) (2^33.18) [P 47.02%][50.00% in 00:00:07][0]
Pub Addr: 3MyCoinoA167kmgPprAidSvv5NoM3Nh6N3
Priv (WIF): p2wpkh-p2sh:L2qvghanHHov914THEzDMTpAyoRmxo7Rh85FLE9oKwYUrycWqudp
Priv (HEX): 0xA7D14FBF43696CA0B3DBFFD0AB7C9ED740FE338B2B856E09F2E681543A444D58
```

```
C:\C++\VanitySearch\x64\Release>VanitySearch.exe -stop -gpu bc1quantum
VanitySearch v1.11
Difficulty: 1073741824
Search: bc1quantum [Compressed]
Start Wed Apr  3 15:01:15 2019
Base Key:B00FD8CDA85B11D4744C09E65C527D35E231D19084FBCA0BF2E48186F31936AE
Number of CPU thread: 7
GPU: GPU #0 GeForce GTX 1050 Ti (6x128 cores) Grid(48x128)
256.896 MK/s (GPU 226.482 MK/s) (2^28.94) [P 38.03%][50.00% in 00:00:00][0]
Pub Addr: bc1quantum898l8mx5pkvq2x250kkqsj7enpx3u4yt
Priv (WIF): p2wpkh:L37xBVcFGeAZ9Tii7igqXBWmfiBhiwwiKQmchNXPV2LNREXQDLCp
Priv (HEX): 0xB00FD8CDA85B11D4744C09E65C527D35E2B1D19095CFCA0BF2E48186F31979C2
```

# Generate a vanity address for a third party using split-key

It is possible to generate a vanity address for a third party in a safe manner using split-key.\
For instance, Alice wants a nice prefix but does not have CPU power. Bob has the requested CPU power but cannot know the private key of Alice, Alice has to use a split-key.

## Step 1

Alice generates a key pair on her computer then send the generated public key and the wanted prefix to Bob. It can be done by email, nothing is secret.  Nevertheless, Alice has to keep safely the private key and not expose it.
```
VanitySearch.exe -s "AliceSeed" -kp
Priv : L4U2Ca2wyo721n7j9nXM9oUWLzCj19nKtLeJuTXZP3AohW9wVgrH
Pub  : 03FC71AE1E88F143E8B05326FC9A83F4DAB93EA88FFEACD37465ED843FCC75AA81
```
Note: The key pair is a standard SecpK1 key pair and can be generated with a third party software.

## Step 2

Bob runs VanitySearch using the Alice's public key and the wanted prefix.
```
VanitySearch.exe -sp 03FC71AE1E88F143E8B05326FC9A83F4DAB93EA88FFEACD37465ED843FCC75AA81 -gpu -stop -o keyinfo.txt 1ALice
```
It generates a keyinfo.txt file containing the partial private key.
```
Pub Addr: 1ALicegohz9YgrLLa4ADCmam7X2Zr6xJZx
PartialPriv: L2hbovuDd8nG4nxjDq1yd5qDsSQiG8xFsAFbHMcThqfjSP6WLg89
```
Bob sends back this file to Alice. It can also be done by email. The partial private key does not allow anyone to guess the final Alice's private key.

## Step 3

Alice can then reconstructs the final private key using her private key (the one generated in step 1) and the keyinfo.txt from Bob.

```
VanitySearch.exe -rp L4U2Ca2wyo721n7j9nXM9oUWLzCj19nKtLeJuTXZP3AohW9wVgrH keyinfo.txt

Pub Addr: 1ALicegohz9YgrLLa4ADCmam7X2Zr6xJZx
Priv (WIF): p2pkh:L1NHFgT826hYNpNN2qd85S7F7cyZTEJ4QQeEinsCFzknt3nj9gqg
Priv (HEX): 0x7BC226A19A1E9770D3B0584FF2CF89E5D43F0DC19076A7DE1943F284DA3FB2D0
```

## How it works

Basically the -sp (start public key) adds the specified starting public key (let's call it Q) to the starting keys of each threads. That means that when you search (using -sp), you do not search for addr(k.G) but for addr(k<sub>part</sub>.G+Q) where k is the private key in the first case and k<sub>part</sub> the "partial private key" in the second case. G is the SecpK1 generator point.\
Then the requester can reconstruct the final private key by doing k<sub>part</sub>+k<sub>secret</sub> (mod n) where k<sub>part</sub> is the partial private key found by the searcher and k<sub>secret</sub> is the private key of Q (Q=k<sub>secret</sub>.G). This is the purpose of the -rp option.\
The searcher has found a match for addr(k<sub>part</sub>.G+k<sub>secret</sub>.G) without knowing k<sub>secret</sub> so the requester has the wanted address addr(k<sub>part</sub>.G+Q) and the corresponding private key k<sub>part</sub>+k<sub>secret</sub> (mod n). The searcher is not able to guess this final private key because he doesn't know k<sub>secret</sub> (he knows only Q).

Note: This explanation is simplified, it does not take care of symmetry and endomorphism optimizations but the idea is the same.

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

