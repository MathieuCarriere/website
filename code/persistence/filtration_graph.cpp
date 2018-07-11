#include <iostream>
#include <set>
#include <fstream>
#include <vector>
#include <string.h>
#include <sstream>
#include <stdlib.h>

using namespace std;

// Warning: assumes vertices are named 0,...,n

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
  int n, m; cin >> n >> m;
  set<simplex, ltstr> Sasc, Sdesc;

  // read vertices
  int id; double f;
  while(n-->0) {
    cin >> id >> f;
    simplex v; v.dim = 0; v.vert.insert(id); v.val = f; vals.push_back(f);
    Sasc.insert(v); if(extended){Sdesc.insert(v);}
  }

  // read edges
  int p, q;
  while(m-->0) {
    cin >> p >> q;
    // add the three edges
    simplex e1; e1.dim = 1; e1.vert.insert(p); e1.vert.insert(q); e1.val = max(vals[p], vals[q]);
    Sasc.insert(e1);
    if(extended){
        simplex e1d; e1d.dim = 1; e1d.vert.insert(p); e1d.vert.insert(q); e1d.val = min(vals[p], vals[q]);
        Sdesc.insert(e1d);
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
