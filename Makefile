#---------------------------------------------------------------------
# Makefile for VanitySearch
#
# Author : Jean-Luc PONS

SRC = Base58.cpp IntGroup.cpp main.cpp Random.cpp \
      Timer.cpp Int.cpp IntMod.cpp Point.cpp SECP256K1.cpp \
      Vanity.cpp GPU/GPUGenerate.cpp hash/ripemd160.cpp \
      hash/sha256.cpp hash/sha512.cpp hash/ripemd160_sse.cpp \
      hash/sha256_sse.cpp

OBJDIR = obj

OBJET = $(addprefix $(OBJDIR)/, \
        Base58.o IntGroup.o main.o Random.o Timer.o Int.o \
        IntMod.o Point.o SECP256K1.o Vanity.o GPU/GPUGenerate.o \
        hash/ripemd160.o hash/sha256.o hash/sha512.o \
        hash/ripemd160_sse.o hash/sha256_sse.o)

CXX        = g++
CUDA       = /usr/local/cuda-8.0
NVCC       = $(CUDA)/bin/nvcc

ifdef gpu
#CXXFLAGS   = -DWIDTHGPU -m64  -Wno-write-strings -g -I. -I$(CUDA)/include
CXXFLAGS   =  -m64  -Wno-write-strings -O2 -I. -I$(CUDA)/include
else
#CXXFLAGS   = -m64  -Wno-write-strings -g -I. -I$(CUDA)/include
CXXFLAGS   =  -m64 -msse -Wno-write-strings -O2 -I. -I$(CUDA)/include
endif

LFLAGS     = -lpthread

#--------------------------------------------------------------------

ifdef gpu
$(OBJDIR)/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -ccbin g++ -m64 -O2 -I$(CUDA)/include -gencode=arch=compute_20,code=sm_20 -o GPU/Engine.o -c GPU/GPUEngine.cu
endif

$(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

all: VanitySearch

VanitySearch: $(OBJET)
	@echo Making VanitySearch...
	$(CXX) $(OBJET) $(LFLAGS) -o VanitySearch

$(OBJET): | $(OBJDIR) $(OBJDIR)/GPU $(OBJDIR)/hash

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/GPU: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p GPU

$(OBJDIR)/hash: $(OBJDIR)
	cd $(OBJDIR) &&	mkdir -p hash

clean:
	@echo Cleaning...
	@rm -f obj/*.o
	@rm -f obj/GPU/*.o
	@rm -f obj/hash/*.o

