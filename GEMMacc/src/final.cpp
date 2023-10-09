#include <bits/stdc++.h>
#include <mmintrin.h>
#include <xmmintrin.h> // SSE
#include <pmmintrin.h> // SSE2
#include <emmintrin.h> // SSE3
#define A(i, j) a[lda * (i) + (j)]
#define B(i, j) b[ldb * (i) + (j)]
#define C(i, j) c[ldc * (i) + (j)]
int m, n, k;
int siz;
/* Block sizes */
#define inc_j 256 // 块的行大小
#define inc_p 128 // 块的列大小
#define min(i, j) ((i) < (j) ? (i) : (j))
using namespace std;
typedef union v4f
{
    __m128d v;
    float s[4];
} v4f_t;

void AddDot4x4(const int k,
               float *a, const int lda,
               float *b, const int ldb,
               float *c, const int ldc);

void gemm(const int m, const int n, const int k,
          float *a, const int lda,
          float *b, const int ldb,
          float *c, const int ldc);

int main(int argc, char *argv[])
{
    m = atoi(argv[1]); // Matrix A 行
    n = atoi(argv[2]); // Matrix B 列
    k = atoi(argv[3]); // Matrix A 列 / Matrix B 行
    // 初始化
    float *a = new float[m * k];
    float *b = new float[k * n];
    float *c = new float[m * n];
    srand(static_cast<unsigned>(time(0)));
    // 随机生成
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            a[i * m + j] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
        }
    }
    for (int i = 0; i < k; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            b[i * k + j] = static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 100.0));
        }
    }
    // 执行gemm并计算过程时间
    auto start = chrono::high_resolution_clock::now();
    int block_k, block_n;
    for (int p = 0; p < k; p += inc_p)
    {
        block_k = min(k - p, inc_p);
        for (int j = 0; j < n; j += inc_j)
        {
            block_n = min(n - j, inc_j);
            int lda = j, ldb = p, ldc = j;
            gemm(m, block_n, block_k, &A(j, p), j, &B(p, 0), p, &C(j, 0), j);
        }
    }
    auto finish = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<chrono::nanoseconds>(finish - start);
    cout << elapsed.count() * 1e-6 << "\n";
    // 清除
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}

void AddDot4x4(const int k,
               float *a, const int lda,
               float *b, const int ldb,
               float *c, const int ldc)
{
    v4f_t
        c_00_01_02_03_vreg,
        c_10_11_12_13_vreg,
        c_20_21_22_23_vreg,
        c_30_31_32_33_vreg,

        b_p0_p1_p2_p3_vreg,

        a_0p_x4_vreg,
        a_1p_x4_vreg,
        a_2p_x4_vreg,
        a_3p_x4_vreg;

    float
        *b_p0_pntr = &B(0, 0);

    c_00_01_02_03_vreg.v = _mm_setzero_pd();
    c_10_11_12_13_vreg.v = _mm_setzero_pd();
    c_20_21_22_23_vreg.v = _mm_setzero_pd();
    c_30_31_32_33_vreg.v = _mm_setzero_pd();

    for (int p = 0; p < k; ++p)
    {
        a_0p_x4_vreg.v = _mm_loaddup_pd((double *)&A(0, p));
        a_1p_x4_vreg.v = _mm_loaddup_pd((double *)&A(1, p));
        a_2p_x4_vreg.v = _mm_loaddup_pd((double *)&A(2, p));
        a_3p_x4_vreg.v = _mm_loaddup_pd((double *)&A(3, p));

        b_p0_pntr += p*n;
        b_p0_p1_p2_p3_vreg.v = _mm_loaddup_pd((double *)b_p0_pntr);
        c_00_01_02_03_vreg.v += a_0p_x4_vreg.v * b_p0_p1_p2_p3_vreg.v;
        c_10_11_12_13_vreg.v += a_1p_x4_vreg.v * b_p0_p1_p2_p3_vreg.v;
        c_20_21_22_23_vreg.v += a_2p_x4_vreg.v * b_p0_p1_p2_p3_vreg.v;
        c_30_31_32_33_vreg.v += a_3p_x4_vreg.v * b_p0_p1_p2_p3_vreg.v;
    }

    C(0, 0) += c_00_01_02_03_vreg.s[0];
    C(0, 1) += c_00_01_02_03_vreg.s[1];
    C(0, 2) += c_00_01_02_03_vreg.s[2];
    C(0, 3) += c_00_01_02_03_vreg.s[3];

    C(1, 0) += c_10_11_12_13_vreg.s[0];
    C(1, 1) += c_10_11_12_13_vreg.s[1];
    C(1, 2) += c_10_11_12_13_vreg.s[2];
    C(1, 3) += c_10_11_12_13_vreg.s[3];

    C(2, 0) += c_20_21_22_23_vreg.s[0];
    C(2, 1) += c_20_21_22_23_vreg.s[1];
    C(2, 2) += c_20_21_22_23_vreg.s[2];
    C(2, 3) += c_20_21_22_23_vreg.s[3];

    C(3, 0) += c_30_31_32_33_vreg.s[0];
    C(3, 1) += c_30_31_32_33_vreg.s[1];
    C(3, 2) += c_30_31_32_33_vreg.s[2];
    C(3, 3) += c_30_31_32_33_vreg.s[3];
}

void gemm(const int m, const int n, const int k,
          float *a, const int lda,
          float *b, const int ldb,
          float *c, const int ldc)
{
    for (int j = 0; j < n; j += 4)
    {
        for (int i = 0; i < m; i += 4)
        {
            float *c_block = &C(i, j);
            AddDot4x4(k, &A(i, 0), lda, &B(0, j), ldb, c_block, ldc);
        }
    }
}
