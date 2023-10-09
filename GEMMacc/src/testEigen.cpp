#include<bits/stdc++.h>
#include<eigen3/Eigen/Dense>
using namespace std;
using Eigen::MatrixXd;
using Eigen::MatrixXf;
int SIZEsq;
int main(int argc,char* argv[]){
    SIZEsq = atoi(argv[1]);
    MatrixXf A = MatrixXf::Random(SIZEsq,SIZEsq);
    MatrixXf B = MatrixXf::Random(SIZEsq,SIZEsq);
    auto start = chrono::high_resolution_clock::now();
    MatrixXf C = A*B;
    auto finish = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(finish-start);
    cout <<elapsed.count()*1e-6 << "\n";
}