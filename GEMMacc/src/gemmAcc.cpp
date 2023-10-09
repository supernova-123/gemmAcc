#include<bits/stdc++.h>
#include<immintrin.h>
using namespace std;
#define SIZE 1024*1024
#define SIZEsqrt 1024
//gemm分块优化
const int blksize = 32;
int M,N,K; // m*k k*n ==> m*n
float A[SIZE+10],B[SIZE+10],C[SIZE+10];
float P1[SIZE+10],P2[SIZE+10];
template<typename T>
inline void gemm_block_bruteforce(T*A, T*B, T*C){
    for(int m = 0;m < M;++m){
        for(int n = 0;n < N;++n){
            T cmn = C[m*N+n];
            for(size_t k = 0;k < K;++k){
                cmn += A[m*K+k]*B[k*N+n];
            }
            C[m*N+n] = cmn;
        }
    }
}

template<typename T>
inline void gemm_block_32(size_t sm, size_t sn, size_t sk, T*A, T*B, T*C){
    for(int m = sm;m < sm+blksize;++m){
        for(int n = sn;n < sn+blksize;++n){
            T cmn = C[m*N+n];
            for(size_t k = sk;k < sk+blksize;++k){
                cmn += A[m*K+k]*B[k*N+n];
            }
            C[m*N+n] = cmn;
        }
    }
}
void gemm_block(){
    for(size_t sm = 0;sm < M;sm += blksize){
        for(size_t sn = 0;sn < N;sn += blksize){
            for(size_t sk = 0;sk < K;sk += blksize){
                gemm_block_32(sm,sn,sk,A,B,C);
            }
        }
    }
}
template<typename T>
inline void gemm_block_32_SIMD(size_t sm, size_t sn, size_t sk, T*A, T*B, T*C){
    size_t m,n,k,x;
    for(m = sm;m<sm+blksize;m++){
        for(n = sn;n<sn+blksize;n+=16){
            __m512 c0;
            //cout << "m = " << m << " n = " << n << "\n";   
            c0 = _mm512_load_ps(C+m*N+n); // 取出连续16个float数据
            
            for(k = sk;k<sk+blksize;k++){
                c0 = _mm512_add_ps(c0, \
				        _mm512_mul_ps( \
                        _mm512_broadcastss_ps( \
                        	_mm_load_ss(A+m*K+k)), \
                     	_mm512_load_ps(B+k*N+n)));
            }
            _mm512_store_ps(C+m*N+n,c0);
        }
    }
}
void gemm_block_SIMD(){
    for(size_t sm = 0;sm < M;sm += blksize){
        for(size_t sn = 0;sn < N;sn += blksize){
            for(size_t sk = 0;sk < K;sk += blksize){
                gemm_block_32_SIMD(sm,sn,sk,A,B,C);
            }
        }
    }
}
template<typename T>
inline void gemm_block_32_FMA(size_t sm, size_t sn, size_t sk, T*A, T*B, T*C){
    size_t m,n,k,x;
    __m512 c0;
    for(m = sm;m<sm+blksize;m++){
        for(n = sn;n<sn+blksize;n+=16){
            c0 = _mm512_load_ps(C+m*N+n); // 取出连续16个float数据
            for(k = sk;k<sk+blksize;k++){
                    c0 = _mm512_fmadd_ps(\
				_mm512_broadcastss_ps(_mm_load_ss(A+m*K+k)), \
                         	_mm512_load_ps(B+k*N+n),c0);
            }
            _mm512_store_ps(C+m*N+n,c0);
        }
    }
}
void gemm_block_FMA(){
    for(size_t sm = 0;sm < M;sm += blksize){
        for(size_t sn = 0;sn < N;sn += blksize){
            for(size_t sk = 0;sk < K;sk += blksize){
                gemm_block_32_FMA(sm,sn,sk,A,B,C);
            }
        }
    }
}
template<typename T>
inline void gemm_block_32_loop(size_t sm, size_t sn, size_t sk, T*A, T*B, T*C){
    size_t m,n,k,x;
    __m512 c0,c1;
    __m512 a0,a1,b0,b1;
    for(m = sm;m<sm+blksize;m++){
        for(n = sn;n<sn+blksize;n+=16*2){
            c0 = _mm512_load_ps(C+m*N+n); // 取出连续16个float数据
            c1 = _mm512_load_ps(C+m*N+n+16);
            for(k = sk;k<sk+blksize;k+=2){
                a0 = _mm512_broadcastss_ps(_mm_load_ss(A+m*K+k));
                a1 = _mm512_broadcastss_ps(_mm_load_ss(A+m*K+k+1));
                b0 = _mm512_load_ps(B+k*N+n);
                b1 = _mm512_load_ps(B+k*N+n+16);
                c0 = _mm512_fmadd_ps(b0,a0,c0);
                c1 = _mm512_fmadd_ps(b1,a0,c1);
                c0 = _mm512_fmadd_ps(b0,a1,c0);
                c1 = _mm512_fmadd_ps(b1,a1,c1);
            }
            _mm512_store_ps(C+m*N+n,c0);
            _mm512_store_ps(C+m*N+n+16,c1);
        }
    }
}
void gemm_block_loop(){
    for(size_t sm = 0;sm < M;sm += blksize){
        for(size_t sn = 0;sn < N;sn += blksize){
            for(size_t sk = 0;sk < K;sk += blksize){
                gemm_block_32_loop(sm,sn,sk,A,B,C);
            }
        }
    }
}

void initABC(){
    for(int i = 0;i < SIZE;++i){
        A[i] = P1[i];
        B[i] = P2[i];
        C[i] = 1.0;
    }
}
int main(){
    srand(static_cast<unsigned>(time(0)));
    M = SIZEsqrt, N = SIZEsqrt, K = SIZEsqrt;
    
    //初始化
    for(int i = 0;i < SIZE;++i){
        P1[i] = static_cast<float>(rand())/(static_cast<float>(RAND_MAX/100.0));
        P2[i] = static_cast<float>(rand())/(static_cast<float>(RAND_MAX/100.0));
    }

    initABC();
    auto start = chrono::high_resolution_clock::now();
    gemm_block_bruteforce(A,B,C);
    auto finish = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(finish-start);
    cout << "暴力计算结果：" << elapsed.count()*1e-6<<" ms" << "\n";

    initABC();
    start = chrono::high_resolution_clock::now();
    gemm_block();
    finish = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(finish-start);
    cout << "分块计算结果：" << elapsed.count()*1e-6<<" ms" << "\n";

    initABC();
    start = chrono::high_resolution_clock::now();
    gemm_block_SIMD(); // -mavx512f
    finish = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(finish-start);
    cout << "SIMD优化计算结果：" << elapsed.count()*1e-6<<" ms" << "\n";

    initABC();
    start = chrono::high_resolution_clock::now();
    gemm_block_FMA(); // -mavx512f
    finish = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(finish-start);
    cout << "FMA优化计算结果：" << elapsed.count()*1e-6<<" ms" << "\n";

    initABC();
    start = chrono::high_resolution_clock::now();
    gemm_block_loop(); // -mavx512f
    finish = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<chrono::nanoseconds>(finish-start);
    cout << "循环展开优化计算结果：" << elapsed.count()*1e-6<<" ms" << "\n";


    
    return 0;
}