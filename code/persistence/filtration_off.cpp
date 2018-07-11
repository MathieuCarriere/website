#include <iostream>
#include <set>
#include <fstream>
#include <vector>
#include <string.h>
#include <sstream>
#include <stdlib.h>

using namespace std;


typedef set<int> vertices;

struct simplex{
  int dim;
  float val;
  vertices vert;
};

struct ltstr
{
  bool operator()(const simplex& i, const simplex& j) const
  {
  if(i.val<j.val) return true;
  else if(i.val==j.val && i.dim<j.dim) return true;
  else if(i.val==j.val && i.dim==j.dim && i.vert < j.vert) return true;
  else return false;
  }
};


int main (int argc, char **argv) {

  if (argc != 2) {
     cout << "Syntax: " << argv[0] << " <0|1>" << endl
     << "   0: ordinary persistence" << endl
     << "   1: extended persistence";
    return 0;
  }

  bool extended = atoi(argv[1]);

  vector<double> vals; vals.clear();

  string line; getline(cin,line); stringstream stream(line); double u;
  while(stream >> u)  vals.push_back(u);
  char buf[256];

  cin.getline(buf, 255);  // skip "OFF"
  int n, m;
  cin >> n >> m;
  cin.getline(buf, 255);  // skip "0"

  set<simplex, ltstr> Sasc, Sdesc;

  // read vertices
  double x,y,z;
  int count = 0;
  while(n-->0) {
    cin >> x >> z >> y;
    simplex v; v.dim = 0; v.vert.insert(count);
    v.val = vals[count]; count++;
    //vals.push_back(z);
    Sasc.insert(v); if(extended){Sdesc.insert(v);}
  }

  // read triangles
  int d, p, q, s;
  while (m-->0) {
    cin >> d >> p >> q >> s;
    // add the three edges
    simplex e1; e1.dim = 1; e1.vert.insert(p); e1.vert.insert(q); e1.val = max(vals[p], vals[q]);
    Sasc.insert(e1);
    if(extended){
        simplex e1d; e1d.dim = 1; e1d.vert.insert(p); e1d.vert.insert(q); e1d.val = min(vals[p], vals[q]);
        Sdesc.insert(e1d);
    }
    simplex e2; e2.dim = 1; e2.vert.insert(p); e2.vert.insert(s); e2.val = max(vals[p], vals[s]);
    Sasc.insert(e2);
    if(extended){
        simplex e2d; e2d.dim = 1; e2d.vert.insert(p); e2d.vert.insert(s); e2d.val = min(vals[p], vals[s]);
        Sdesc.insert(e2d);
    }
    simplex e3; e3.dim = 1; e3.vert.insert(q); e3.vert.insert(s); e3.val = max(vals[q], vals[s]);
    Sasc.insert(e3);
    if(extended){
        simplex e3d; e3d.dim = 1; e3d.vert.insert(q); e3d.vert.insert(s); e3d.val = min(vals[q], vals[s]);
        Sdesc.insert(e3d);
    }
    // add the triangle
    simplex t; t.dim = 2; t.vert.insert(p); t.vert.insert(q); t.vert.insert(s); t.val = max(vals[p], max(vals[q], vals[s]));
    Sasc.insert(t);
    if(extended){
        simplex td; td.dim = 2; td.vert.insert(p); td.vert.insert(q); td.vert.insert(s); td.val = min(vals[p], min(vals[q], vals[s]));
        Sdesc.insert(td);
    }
  }

  // output filtration
  for (set<simplex, ltstr>::iterator sit = Sasc.begin(); sit != Sasc.end(); sit++) {
    cout << sit->val << " " << sit->dim << " ";
    for (vertices::iterator vit = sit->vert.begin(); vit != sit->vert.end(); vit++)
      cout << *vit << " ";
    cout << endl;
  }
  if(extended){
    for (set<simplex, ltstr>::iterator sit = Sdesc.begin(); sit != Sdesc.end(); sit++) {
      cout << sit->val << " " << sit->dim << " ";
      for (vertices::iterator vit = sit->vert.begin(); vit != sit->vert.end(); vit++)
        cout << *vit << " ";
      cout << endl;
      }

  }

  return 0;

}
