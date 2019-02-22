#include "Point.h"

Point::Point() {
}

Point::Point(const Point &p) {
  x.Set((Int *)&p.x);
  y.Set((Int *)&p.y);
  z.Set((Int *)&p.z);
}

Point::Point(Int *cx,Int *cy,Int *cz) {
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::Point(Int *cx, Int *cz) {
  x.Set(cx);
  z.Set(cz);
}

void Point::Clear() {
  x.SetInt32(0);
  y.SetInt32(0);
  z.SetInt32(0);
}

void Point::Set(Int *cx, Int *cy,Int *cz) {
  x.Set(cx);
  y.Set(cy);
  z.Set(cz);
}

Point::~Point() {
}

void Point::Set(Point &p) {
  x.Set(&p.x);
  y.Set(&p.y);
}

bool Point::isZero() {
  return x.IsZero() && y.IsZero();
}

void Point::Reduce() {

  Int i(&z);
  i.ModInv();
  x.ModMul(&x,&i);
  y.ModMul(&y,&i);
  z.SetInt32(1);

}

bool Point::equals(Point &p) {
  return x.IsEqual(&p.x) && y.IsEqual(&p.y) && z.IsEqual(&p.z);
}

std::string Point::toString() {

  std::string ret;
  ret  = "X=" + x.GetBase16() + "\n";
  ret += "Y=" + y.GetBase16() + "\n";
  ret += "Z=" + z.GetBase16() + "\n";
  return ret;

}
