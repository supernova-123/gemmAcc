#include<bits/stdc++.h>
#include<cblas.h>
using namespace std;
int main(int argc, char* argv[]){
    srand(static_cast<unsigned>(time(0)));
    int siz = atoi(argv[1]);
    int m = siz, n = siz;
    double* A = new double[siz*siz+10], *B = new double[siz*siz+10];
    double *C = new double[siz*siz+10];
    for(int i = 0;i < siz*siz;++i){
        A[i] = static_cast<float>(rand())/(static_cast<float>(RAND_MAX/100.0));
        B[i] = static_cast<float>(rand())/(static_cast<float>(RAND_MAX/100.0));
    }
    auto start = chrono::high_resolution_clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m,n,n,1.0,A,n,B,n,0.0,C,n);
    auto finish = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(finish-start);
    cout << elapsed.count()*1e-6<< "\n";
    return 0;
}