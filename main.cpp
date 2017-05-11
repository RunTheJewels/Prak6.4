#include <iostream>
#include <cmath>
#include <fstream>
#include <complex>
#include <algorithm>
#include <vector>
#include <string>
#include "mpi.h"
#include <omp.h>


using namespace std;
typedef std::complex<float> complexd;
uint threads;

void cubit(vector<complexd>& a, vector<complexd>& b, int loc_size, int N, int K1, int K2, complexd H[4][4]);


int rank = 0, comm_size;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

    try
    {
        MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (argc != 5)
            throw string("Wrong arguments");

        uint N = atoi(argv[1]);
        uint K1 = atoi(argv[2]);
        uint K2 = atoi(argv[3]);
        threads = atoi(argv[4]);

        uint vec_size = (uint) pow((float) 2, (float) N);
        if (vec_size % comm_size != 0 || (uint) comm_size > vec_size)
            throw string("Wrong number of processors");

        uint loc_size = vec_size / comm_size;

        int seed = time(0);
        double start_time, end_time, time_diff_comp = 0;
        double all_times_comp[comm_size];

        vector<complexd > a(loc_size), b(loc_size);

        MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
        srand(seed + rank);

        float sum = 0.0;
        float all_sum = 0.0;
        for (uint i = 0; i < loc_size; ++i)
        {
            a[i] = complexd((float) std::rand()/RAND_MAX, (float) std::rand()/RAND_MAX);
            sum += std::abs(a[i])*std::abs(a[i]);
        }

        MPI_Allreduce(&all_sum, &sum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        all_sum = std::sqrt(all_sum);
        for (uint i = 0; i < loc_size; ++i)
        {
            a[i] /= all_sum;
        }

        vector<complexd> all_res(vec_size*(rank==0));

        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();

        complexd H[4][4];
        for (int i = 0; i < 16; i++)
        {
            H[i / 4][i % 4] = 0;
        }

        H[0][0] = H[1][1] = H[2][3] = H[3][2] = 1;


        cubit(a, b, loc_size, N, K1, K2, H);

        end_time = MPI_Wtime();
        time_diff_comp = end_time - start_time;

        MPI_Gather(b.data(),loc_size,MPI_COMPLEX,all_res.data(),loc_size,MPI_COMPLEX,0,MPI_COMM_WORLD);

        MPI_Gather(&time_diff_comp, 1, MPI_DOUBLE, all_times_comp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            cout << comm_size << " " << threads << " " << N << " " << K1 << " "
            << K2 << " " << *std::max_element(&all_times_comp[0], &all_times_comp[comm_size]) << endl;
        }
    }
    catch (const string& e) {
        if (rank == 0)
        cerr << e << endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    catch (const MPI::Exception& e) {
        if (rank == 0)
        //cerr << e.Get_error_code() << endl;
        cerr << "MPI Exception thrown" << endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    MPI_Finalize();
    return 0;
}

void cubit(vector<complexd>& a, vector<complexd>& b, int loc_size, int N, int K1, int K2, complexd H[4][4])
{
    int P1 = N - K1;
    int stride1 = 1 << P1;
    int P2 = N - K2;
    int stride2 = 1 << P2;

    if (stride1 < loc_size and stride2 < loc_size)
    {
        // All elements are here in a

        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < loc_size; ++i)
        {
            int i00 = i & ~stride1 & ~stride2;
            int i01 = i & ~stride1 | stride2;
            int i10 = (i | stride1) & ~stride2;
            int i11 = i | stride1 | stride2;
            int ik1 = (i & stride1) >> P1;
            int ik2 = (i & stride2) >> P2;
            int ik = (ik1 << 1) + ik2;
            b[i] = H[ik][0] * a[i00] + H[ik][1] * a[i01] +
            H[ik][2] * a[i10] + H[ik][3] * a[i11];
        }
    } else if (stride1 < loc_size)
    {
        int proc_stride = stride1 / loc_size;
        vector<complexd> tmp(loc_size);

        // MPI_Isend(a.data(),loc_size,MPI::COMPLEX,proc_stride^rank,0,MPI_COMM_WORLD);
        //
        // MPI_Recv(tmp.data(),loc_size,MPI::COMPLEX,proc_stride^rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        MPI_Sendrecv(a.data(),loc_size,MPI::COMPLEX,proc_stride^rank,0,tmp.data(),
        loc_size,MPI::COMPLEX,proc_stride^rank,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < loc_size; ++i)
        {
            int i00 = i & ~stride1 & ~stride2;
            int i01 = i & ~stride1;
            int i10 = (i | stride1) & ~stride2;
            int i11 = i | stride1;
            int ik1 = (i & stride1) >> P1;
            int ik2 = (i & stride2) >> P2;
            int ik = (ik1 << 1) + ik2;

            b[i] = H[ik][0] * a[i00] + H[ik][1] * tmp[i01] +
            H[ik][2] * a[i10] + H[ik][3] * tmp[i11];

        }
    } else if (stride2 < loc_size)
    {
        int proc_stride = stride2 / loc_size;
        vector<complexd> tmp(loc_size);

        // MPI_Isend(a.data(),loc_size,MPI::COMPLEX,proc_stride^rank,0,MPI_COMM_WORLD);
        //
        // MPI_Recv(tmp.data(),loc_size,MPI::COMPLEX,proc_stride^rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        MPI_Sendrecv(a.data(),loc_size,MPI::COMPLEX,proc_stride^rank,0,tmp.data(),
        loc_size,MPI::COMPLEX,proc_stride^rank,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < loc_size; ++i)
        {
            int i00 = i & ~stride1 & ~stride2;
            int i01 = i & ~stride1 | stride2;
            int i10 = i & ~stride2;
            int i11 = i | stride2;
            int ik1 = (i & stride1) >> P1;
            int ik2 = (i & stride2) >> P2;
            int ik = (ik1 << 1) + ik2;

            b[i] = H[ik][0] * a[i00] + H[ik][1] * a[i01] +
            H[ik][2] * tmp[i10] + H[ik][3] * tmp[i11];

        }
    } else
    {
        int proc_stride1 = stride1 / loc_size;
        int proc_stride2 = stride2 / loc_size;
        vector<complexd> tmp1(loc_size),tmp2(loc_size),tmp3(loc_size);

        // MPI_Isend(a.data(),loc_size,MPI::COMPLEX,proc_stride1^rank,0,MPI_COMM_WORLD);
        //
        // MPI_Recv(tmp1.data(),loc_size,MPI::COMPLEX,proc_stride1^rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        MPI_Sendrecv(a.data(),loc_size,MPI::COMPLEX,proc_stride1^rank,0,tmp1.data(),
        loc_size,MPI::COMPLEX,proc_stride1^rank,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // MPI_Isend(a.data(),loc_size,MPI::COMPLEX,proc_stride2^rank,0,MPI_COMM_WORLD);
        //
        // MPI_Recv(tmp2.data(),loc_size,MPI::COMPLEX,proc_stride2^rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        MPI_Sendrecv(a.data(),loc_size,MPI::COMPLEX,proc_stride2^rank,0,tmp2.data(),
        loc_size,MPI::COMPLEX,proc_stride2^rank,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // MPI_Isend(a.data(),loc_size,MPI::COMPLEX,proc_stride1^proc_stride2^rank,0,MPI_COMM_WORLD);
        //
        // MPI_Recv(tmp3.data(),loc_size,MPI::COMPLEX,proc_stride1^proc_stride2^rank,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

        MPI_Sendrecv(a.data(),loc_size,MPI::COMPLEX,proc_stride1^proc_stride2^rank,0,tmp3.data(),
        loc_size,MPI::COMPLEX,proc_stride1^proc_stride2^rank,0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        #pragma omp parallel for num_threads(threads)
        for (int i = 0; i < loc_size; ++i)
        {
            int i00 = i & ~stride1 & ~stride2;
            int i01 = i & ~stride1;
            int i10 = i & ~stride2;
            int i11 = i;
            int ik1 = (i & stride1) >> P1;
            int ik2 = (i & stride2) >> P2;
            int ik = (ik1 << 1) + ik2;
            b[i] = H[ik][0] * a[i00] + H[ik][1] * tmp2[i01] +
            H[ik][2] * tmp1[i10] + H[ik][3] * tmp3[i11];

        }
    }
}
