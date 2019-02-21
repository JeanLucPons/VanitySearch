#include "IntGroup.h"

using namespace std;

IntGroup::IntGroup() {
  subp = (Int *)malloc(CPU_GRP_SIZE * sizeof(Int));
}

IntGroup::~IntGroup() {
  free(subp);
}

void IntGroup::Set(Int *pts) {
  ints = pts;
}

// Compute modular inversion of the whole group
void IntGroup::ModInv() {

  Int newValue;
  Int inverse;

  subp[0].Set(&ints[0]);
  for (unsigned int i = 1; i < CPU_GRP_SIZE; i++) {
    subp[i].MontgomeryMult(&subp[i - 1], &ints[i]);
  }

  // Do the inversion
  inverse.Set(&subp[CPU_GRP_SIZE - 1]);
  inverse.ModInv();

  for (int i = CPU_GRP_SIZE - 1; i > 0; i--) {
    newValue.MontgomeryMult(&subp[i - 1], &inverse);
    inverse.MontgomeryMult(&ints[i]);
    ints[i].Set(&newValue);
  }

  ints[0].Set(&inverse);

}