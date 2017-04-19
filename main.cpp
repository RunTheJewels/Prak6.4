#include <complex>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>

typedef std::complex<float> complexd;

using namespace std;

int kth_bit(int k)
{
    return 1 << (k-1);
}

int ex2(int n)
{
    int r = 1;
    for (int i = 0; i < n; i++)
    {
        r*=2;
    }
    return r;
}

int main(int argc, char** argv)
{
    int n,k1,k2,threads;
    complexd *a,*b,u[4][4];
    long long int N;
    unsigned seed;
    float arg;

    if (argc < 4)
    {
        cout << "Недостаточно аргументов";
        return -1;
    }
    n = atoi(argv[1]);
    k1 = atoi(argv[2]);
    k2 = atoi(argv[3]);
    threads = atoi(argv[4]);

    for (int i = 0; i < 16; i++)
    {
        u[i / 4][i % 4] = 0;
    }

    u[0][0] = u[1][1] = u[2][3] = u[3][2] = 1;

    N = 1 << n;
    cout << N;
    a = new complexd [N];
    b = new complexd [N];

    #pragma omp parallel num_threads(threads) private(seed,arg) shared(a)
    {
        seed = omp_get_thread_num()+time(NULL);
        #pragma omp for
        for (int i = 0; i < N; i++)
        {
            arg = (float)rand_r(&seed);
            a[i] = complex <float> (cos(arg),sin(arg));
        }
    }

    double start=omp_get_wtime();


    int shift1 = n - k1;
    int shift2 = n - k2;
    //Все биты нулевые, за исключением соответсвующего номеру первого изменяемого кубита
    int pow2k1 = 1 << (shift1);
    //Все биты нулевые, за исключением соответсвующего номеру второгоизменяемого кубита
    int pow2k2 = 1 << (shift2);

    #pragma omp parallel for shared(a,b)
    for	(int i = 0; i < N; i++)
    {
        //Установка изменяемых битов во все возможные позиции
        int i00 = i & ~pow2k1 & ~pow2k1;
        int i01 = i & ~pow2k1 | pow2k1;
        int i10 = (i | pow2k1) & ~pow2k1;
        int i11 = i | pow2k1 | pow2k1;
        //Получение значений изменяемых битов
        int ik1 = (i & pow2k1) >> shift1;
        int ik2 = (i & pow2k1) >> shift2;
        //Индекс в матрице
        int ik = (ik1 << 1) + ik2;
        b[i] = u[ik][0] * a[i00] + u[ik][1] * a[i01] +
            u[ik][2] * a[i10] + u[ik][3] * a[i11];
    }

    double finish=omp_get_wtime();

    // cout << endl;
    // for (int i = 0; i < N; i++)
    // {
    //     cout << b[i] << " ";
    // }

    delete []a;
    delete []b;

    FILE *f = fopen("./data.txt", "a");
    fprintf(f,"%d %d %d %d %f\n",threads,n,k1,k2,finish-start);
    fclose(f);

    return 0;
}
