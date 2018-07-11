#include <cstdlib>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <set>
#include <map>
#include <limits>
#include <cmath>
#include <cfloat>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/casts.hpp>
#include <boost/function/function_base.hpp>


using namespace std;

typedef vector<int> boundary;
typedef set<int> vertices;
typedef vector<boundary> boundary_matrix;
typedef vector<pair<int, pair<float, float> > > barcode;
typedef vector<pair<pair<int,int>, pair<float, float> > > ext_barcode;

struct simplex{
  int dim;
  float val;
  vertices vert;
};

bool comp_filter (simplex i, simplex j) { 
  if(i.val<j.val){return true;}
  else{
    if(i.val==j.val){return (i.dim<j.dim);}
    else{return false;}
  }
}

bool comp_filter_d (simplex i, simplex j) {
  if(i.val>j.val){return true;}
  else{
    if(i.val==j.val){return (i.dim<j.dim);}
    else{return false;}
  }
}

bool mycomp (int i, int j){return (i>j);}

bool comp_pers (pair<int, pair<float, float> > i, pair<int, pair<float, float> > j) {
  return (abs(i.second.second-i.second.first) > abs(j.second.second-j.second.first));
}

bool comp_ext_pers (pair<pair<int,int>, pair<float, float> > i, pair<pair<int,int>, pair<float, float> > j) {
  return (abs(i.second.second-i.second.first) > abs(j.second.second-j.second.first));
}


boundary add_in_Z2(boundary L1, boundary L2){
  boundary T; T.clear(); int l = L1.size();
  for(int i = 0; i < l; i++){
    if (L1[i] == L2[i])
      T.push_back(0);
    else
      T.push_back(1);
  }
  return T;
}

int lowest_row(boundary L){
  int low = -1; int l = L.size();
  for(int i = 0; i < l; i++)
    if(L[i] != 0)
      low = i;
  return low;
}

boundary sparse_add_in_Z2(boundary L1, boundary L2){
  int i = 0; int j = 0; int l1 = L1.size(); int l2 = L2.size();
  boundary T; T.clear();
  while( i < l1 and j < l2){
    if(L1[i] > L2[j]){T.push_back(L1[i]); i += 1;}
    else if(L2[j] > L1[i]){T.push_back(L2[j]); j += 1;}
    else{i += 1; j += 1;}  // L1[j] == L2[i]
  }
  if(i == l1){
    while(j < l2){T.push_back(L2[j]); j += 1;}
  }
  if(j == l2){
    while(i < l1){T.push_back(L1[i]); i += 1;}
  }
  return T;
}


vector<simplex> read_filtration(){

  vector<simplex> F; F.clear();
  string line;
  while(getline(cin,line)){
    stringstream stream(line);
    simplex s; s.vert.clear();
    s.dim = -1; stream >> s.val; stream >> s.dim; int i = 0;
    while(i <= s.dim){
      int f; stream >> f;
      s.vert.insert(f); i++;
    }
    if(s.dim != -1)
      F.push_back(s);
  }

  return F;

}















boundary_matrix compute_ordinary_boundary_matrix(vector<simplex> & F){

  boundary_matrix B; B.clear();
  int num_simplex = F.size();

  // sort filtration by function values and dimensions
  sort(F.begin(), F.end(), comp_filter);

  // re-sort now by lexicographical order
  map<vertices,int> lexico;
  for (int i = 0; i < num_simplex; i++)
    lexico.insert(pair<vertices,int> (F[i].vert, i));

  // create sparse matrix
  for (int i = 0; i < num_simplex; ++i ){
    boundary h; vertices b = F[i].vert; int num_simplex_bound = b.size();
    // compute simplices of codimension 1
    for(int j = 0; j < num_simplex_bound; j++){
      vertices bb = b;
      vertices::iterator vit = bb.begin();
      for (int k = 0; k < j; k++)
        vit++;
      bb.erase(vit);
      // find them in the filtration
      map<vertices,int>::iterator sit = lexico.find(bb);
      if(sit != lexico.end())
        h.push_back(sit->second);
    }
    // simplices in the boundary are assumed to be sorted in decreasing order
    sort(h.begin(), h.end(), mycomp);
    B.push_back(h);
  }

  return B;
}












pair<boundary_matrix, pair<vector<simplex>, vector<simplex> > > compute_extended_boundary_matrix(vector<simplex> & F){
  
  boundary_matrix B; B.clear(); int num_simplex;
  num_simplex = F.size()/2;
  vector<simplex> Fasc, Fdesc; Fasc.clear(); Fdesc.clear();
  for(int i = 0; i < num_simplex; i++)
    Fasc.push_back(F[i]);
  for(int i = 0; i < num_simplex; i++)
    Fdesc.push_back(F[num_simplex+i]);

  // sort filtration by function values and dimensions
  sort(Fasc.begin(), Fasc.end(), comp_filter);
  sort(Fdesc.begin(), Fdesc.end(), comp_filter_d);

  // re-sort now by lexicographical order
  map<vertices,int> lexico, lexicodesc;
  for (int i = 0; i < num_simplex; i++)
    lexico.insert(pair<vertices,int> (Fasc[i].vert, i));
  for (int i = 0; i < num_simplex; i++)
    lexicodesc.insert(pair<vertices,int> (Fdesc[i].vert, i));

  // create sparse matrix

  // ascending filtration
  for (int i = 0; i < num_simplex; ++i ){
    boundary h; vertices b = Fasc[i].vert; int num_simplex_bound = b.size();
    // compute simplices of codimension 1
    for(int j = 0; j < num_simplex_bound; j++){
      vertices bb = b;
      vertices::iterator vit = bb.begin();
      for (int k = 0; k < j; k++)
        vit++;
      bb.erase(vit);
      // find them in the filtration
      map<vertices,int>::iterator sit = lexico.find(bb);
      if(sit != lexico.end())
        h.push_back(sit->second);
    }
    // simplices in the boundary are assumed to be sorted in decreasing order
    sort(h.begin(), h.end(), mycomp);
    B.push_back(h);
  }

  //descending filtration
  for (int i = 0; i < num_simplex; ++i ){
    boundary h; vertices b = Fdesc[i].vert; int num_simplex_bound = b.size();
    map<vertices,int>::iterator sit = lexico.find(b);
    if(sit != lexico.end())
      h.push_back(sit->second);
    // compute simplices of codimension 1
    for(int j = 0; j < num_simplex_bound; j++){
      vertices bb = b;
      vertices::iterator vit = bb.begin();
      for (int k = 0; k < j; k++)
        vit++;
      bb.erase(vit);
      // find them in the filtration
      map<vertices,int>::iterator sit = lexicodesc.find(bb);
      if(sit != lexicodesc.end())
        h.push_back(num_simplex+sit->second);
    }
    // simplices in the boundary are assumed to be sorted in decreasing order
    sort(h.begin(), h.end(), mycomp);
    B.push_back(h);
  }

  pair<vector<simplex>, vector<simplex> > V(Fasc,Fdesc);
  pair<boundary_matrix, pair<vector<simplex>, vector<simplex> > >P(B,V);

  return P;

}












vector<simplex> G;
struct sort_pred {
  bool operator()(const pair<int,int> &left, const pair<int,int> &right) {
        return abs(G[left.second].val-G[left.first].val) > abs(G[right.second].val-G[right.first].val);
    }
};

barcode compute_ordinary_barcode_sparse(boundary_matrix B, vector<simplex> & F){

  // initialization
  boundary_matrix R = B;
  boundary L; L.clear();
  int num_simplex = B.size();
  vector< pair<int, int> > bc_int;
  barcode bc; bc.clear();
  G=F;


  // algorithm
  for(int j = 0; j < num_simplex; j++){
    pair<int, int> p(j,j); bc_int.push_back(p);
    L = B[j]; R[j].clear();
    while( L.size() > 0 and R[L[0]].size() > 0 ){
      L[0];
      L = sparse_add_in_Z2(L,R[L[0]]);
    }
    if( L.size() > 0 ){
      R[L[0]] = L;
      bc_int[L[0]].second = j;
    }
    else
      bc_int[j].second = -1;
  }

  //sort(bc_int.begin(),bc_int.end(),sort_pred());

  // function values & dimensions
  for(int i = 0; i < bc_int.size(); i++){
    int d = F[bc_int[i].first].dim;
    if(bc_int[i].second != -1 && F[bc_int[i].first].val != F[bc_int[i].second].val){
      pair<float, float> p(F[bc_int[i].first].val, F[bc_int[i].second].val);
      //cout << d << " " << *(F[bc_int[i].first].vert.begin()) << " " << *(F[bc_int[i].second].vert.begin()) << endl;
      pair<int, pair<float, float> > q(d,p);
      bc.push_back(q);
    }
    if(bc_int[i].second == -1){
      pair<float, float> p(F[bc_int[i].first].val, FLT_MAX/*numeric_limits<float>::infinity()*/);
      int d = F[bc_int[i].first].dim;
      pair<int, pair<float, float> > q(d,p);
      bc.push_back(q);
    }
  }

  return bc;

}

















ext_barcode compute_extended_barcode_sparse(boundary_matrix B, const vector<simplex> & Fasc, const vector<simplex> & Fdesc){

  // initialization
  boundary_matrix R = B;
  boundary L; L.clear();
  int num_simplex = B.size();
  vector< pair<int, int> > bc_int;
  ext_barcode bc; bc.clear();

  // algorithm
  for(int j = 0; j < num_simplex; j++){
    pair<int, int> p(j,j); bc_int.push_back(p);
    L = B[j]; R[j].clear();
    while( L.size() > 0 and R[L[0]].size() > 0 ){
      L[0]; 
      L = sparse_add_in_Z2(L,R[L[0]]);
    }
    if( L.size() > 0 ){
      R[L[0]] = L; 
      bc_int[L[0]].second = j;
    }
    else
      bc_int[j].second = -1;
  }

  // function values & dimensions
  for(int i = 0; i < bc_int.size(); i++){

    double f,g; int type;
    if (bc_int[i].first < num_simplex/2 && bc_int[i].second < num_simplex/2){
      f = Fasc[bc_int[i].first].val; g = Fasc[bc_int[i].second].val; type = 0;}
    if (bc_int[i].first < num_simplex/2 && bc_int[i].second >= num_simplex/2){
      f = Fasc[bc_int[i].first].val; g = Fdesc[bc_int[i].second-num_simplex/2].val; type = 1;}
    if (bc_int[i].first >= num_simplex/2 && bc_int[i].second >= num_simplex/2){
      f = Fdesc[bc_int[i].first-num_simplex/2].val; g = Fdesc[bc_int[i].second-num_simplex/2].val; type = 2;}

    int d;
    if (bc_int[i].first < num_simplex/2)
        d = Fasc[bc_int[i].first].dim;
    else
        d = Fdesc[bc_int[i].first-num_simplex/2].dim + 1;

    if(f != g){
      pair<float, float> p(f,g); pair<int,int> pi(d,type);
      pair<pair<int,int>, pair<float, float> > q(pi,p);
      bc.push_back(q);
    }
  }

  return bc;

}
















int main(int argc, char** argv) {

  if (argc != 2) {
    cout << "Syntax: " << argv[0] << " <0|1>" << endl
         << "   0: ordinary persistence" << endl
         << "   1: extended persistence";
    return 0;
  }
  
  bool extended = atoi(argv[1]);
  vector<simplex> F = read_filtration();

  if(extended){
    pair<boundary_matrix, pair<vector<simplex>, vector<simplex> > > P = compute_extended_boundary_matrix(F);
    boundary_matrix B = P.first; vector<simplex> Fasc = P.second.first; vector<simplex> Fdesc = P.second.second;
    ext_barcode b = compute_extended_barcode_sparse(B,Fasc,Fdesc);
    sort(b.begin(), b.end(), comp_ext_pers);
    for(int i = 0 ; i < b.size(); i++){
      if (b[i].first.second == 0)
        cout << b[i].second.first << " " << b[i].second.second << " " << b[i].first.first << /*" " << "ord" <<*/ endl;
      if (b[i].first.second == 1)
        cout << b[i].second.first << " " << b[i].second.second << " " << b[i].first.first << /*" " << "ext" <<*/ endl;
      if (b[i].first.second == 2)
        cout << b[i].second.first << " " << b[i].second.second << " " << b[i].first.first << /*" " << "rel" <<*/ endl;
    }
  }
  else{
    boundary_matrix B = compute_ordinary_boundary_matrix(F);
    barcode b = compute_ordinary_barcode_sparse(B,F);
    sort(b.begin(), b.end(), comp_pers);
    for(int i = 0 ; i < b.size(); i++)
      cout << b[i].second.first << " " << b[i].second.second << " " << b[i].first << endl;
  }
  

return 0;
}
