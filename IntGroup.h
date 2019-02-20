#ifndef INTGROUPH
#define INTGROUPH

#include "Int.h"
#include <vector>

#define CPU_GRP_SIZE 256

class IntGroup {

public:

	IntGroup();
	~IntGroup();
	void Set(Int *pts);
	void ModInv();

private:

	Int *ints;
  Int *subp;
  int log2n;                  // log2(n)

};

#endif // INTGROUPCPUH
