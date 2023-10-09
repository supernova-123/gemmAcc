#include<bits/stdc++.h>
#include <armadillo>
using namespace std;
using namespace arma;
int SIZEsqrt;
// -larmadillo
int main(int argc, char* argv[])
{
    SIZEsqrt = atoi(argv[1]);
    int M = SIZEsqrt, N = SIZEsqrt;
    vector<float> P1(SIZEsqrt*SIZEsqrt+10,0),P2(SIZEsqrt*SIZEsqrt+10,0);
    for(int i = 0;i < SIZEsqrt*SIZEsqrt;++i){
        P1[i] = static_cast<float>(rand())/(static_cast<float>(RAND_MAX/100.0));
        P2[i] = static_cast<float>(rand())/(static_cast<float>(RAND_MAX/100.0));
    }
    // armadillo库
    mat X(SIZEsqrt, SIZEsqrt), Y(SIZEsqrt, SIZEsqrt);
    for(int i = 0;i < M;++i){
        for(int j = 0;j < N;++j){
            X(i,j) = P1[i*N+j];
            Y(i,j) = P2[i*N+j];
        }
    }
    auto start = chrono::high_resolution_clock::now();
    mat AmulB = X*Y;
    auto finish = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(finish-start);
    cout << elapsed.count()*1e-6<< "\n";
    return 0;
}