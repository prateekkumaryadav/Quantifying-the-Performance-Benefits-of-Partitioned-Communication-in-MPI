#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <omp.h>

#define WARMUP 15
#define ITERS 100

void delay_us(double us) {
    if (us <= 0.0) return;
    double start = MPI_Wtime();
    while ((MPI_Wtime() - start) * 1e6 < us) {}
}

// 1. Pt2Pt Single (Bulk Sync - Figure 1)
double bench_pt2pt_single(int rank, int num_threads, int theta, size_t part_size, double gamma_us_per_mb) {
    int total_parts = num_threads * theta;
    size_t total_size = part_size * total_parts;
    char *buf = (char *)calloc(total_size, 1);
    MPI_Request req;

    if (rank == 0) {
        MPI_Send_init(buf, (int)total_size, MPI_BYTE, 1, 100, MPI_COMM_WORLD, &req);
    } else {
        MPI_Recv_init(buf, (int)total_size, MPI_BYTE, 0, 100, MPI_COMM_WORLD, &req);
    }

    double delay_time_us = gamma_us_per_mb * ((double)part_size / (1024.0 * 1024.0));
    double total_comm_time = 0.0;

    #pragma omp parallel num_threads(num_threads) shared(req, total_comm_time)
    {
        int tid = omp_get_thread_num();
        for (int iter = -WARMUP; iter < ITERS; iter++) {
            #pragma omp barrier
            #pragma omp master
            MPI_Barrier(MPI_COMM_WORLD);
            #pragma omp barrier

            double t_start = MPI_Wtime();

            if (rank == 0 && tid == num_threads - 1) {
                delay_us(delay_time_us);
            }

            // Single bulk synchronization before master sends (Figure 1)
            #pragma omp barrier
            #pragma omp master
            {
                MPI_Start(&req);
                MPI_Wait(&req, MPI_STATUS_IGNORE);
            }

            double t_end = MPI_Wtime();
            if (iter >= 0 && tid == 0) {
                double comm_only = (t_end - t_start) - (delay_time_us * 1e-6);
                if (comm_only < 0.0) comm_only = t_end - t_start;
                total_comm_time += comm_only;
            }
        }
    }

    MPI_Request_free(&req);
    free(buf);
    return (total_comm_time / ITERS) * 1e6;
}

// 2. Pt2Pt Many (Pipelined with per-thread MPI_Comm_dup - Figure 2)
double bench_pt2pt_many(int rank, int num_threads, int theta, size_t part_size, double gamma_us_per_mb) {
    int total_parts = num_threads * theta;
    size_t total_size = part_size * total_parts;
    char *buf = (char *)calloc(total_size, 1);
    MPI_Comm *comms = (MPI_Comm *)malloc(num_threads * sizeof(MPI_Comm));
    MPI_Request *reqs = (MPI_Request *)malloc(total_parts * sizeof(MPI_Request));

    for (int t = 0; t < num_threads; t++) {
        MPI_Comm_dup(MPI_COMM_WORLD, &comms[t]);
        for (int p = 0; p < theta; p++) {
            int idx = t * theta + p;
            if (rank == 0) {
                MPI_Send_init(buf + idx * part_size, (int)part_size, MPI_BYTE, 1, 200 + idx, comms[t], &reqs[idx]);
            } else {
                MPI_Recv_init(buf + idx * part_size, (int)part_size, MPI_BYTE, 0, 200 + idx, comms[t], &reqs[idx]);
            }
        }
    }

    double delay_time_us = gamma_us_per_mb * ((double)part_size / (1024.0 * 1024.0));
    double total_comm_time = 0.0;

    #pragma omp parallel num_threads(num_threads) shared(reqs, total_comm_time)
    {
        int tid = omp_get_thread_num();
        for (int iter = -WARMUP; iter < ITERS; iter++) {
            #pragma omp barrier
            #pragma omp master
            MPI_Barrier(MPI_COMM_WORLD);
            #pragma omp barrier

            double t_start = MPI_Wtime();

            if (rank == 0) {
                if (tid == num_threads - 1) {
                    delay_us(delay_time_us);
                }
                for (int p = 0; p < theta; p++) {
                    int idx = tid * theta + p;
                    MPI_Start(&reqs[idx]);
                    MPI_Wait(&reqs[idx], MPI_STATUS_IGNORE);
                }
            } else {
                for (int p = 0; p < theta; p++) {
                    int idx = tid * theta + p;
                    MPI_Start(&reqs[idx]);
                    MPI_Wait(&reqs[idx], MPI_STATUS_IGNORE);
                }
            }

            #pragma omp barrier
            double t_end = MPI_Wtime();

            if (iter >= 0 && tid == 0) {
                double comm_only = (t_end - t_start) - (delay_time_us * 1e-6);
                if (comm_only < 0.0) comm_only = t_end - t_start;
                total_comm_time += comm_only;
            }
        }
    }

    for (int idx = 0; idx < total_parts; idx++) MPI_Request_free(&reqs[idx]);
    for (int t = 0; t < num_threads; t++) MPI_Comm_free(&comms[t]);
    free(comms);
    free(reqs);
    free(buf);
    return (total_comm_time / ITERS) * 1e6;
}

// 3. Pt2Pt Partitioned (MPI 4.0 Partitioned API - Figure 3)
double bench_pt2pt_part(int rank, int num_threads, int theta, size_t part_size, double gamma_us_per_mb) {
    int total_parts = num_threads * theta;
    size_t total_size = part_size * total_parts;
    char *buf = (char *)calloc(total_size, 1);
    MPI_Request req;

    if (rank == 0) {
        MPI_Psend_init(buf, total_parts, (MPI_Count)part_size, MPI_BYTE, 1, 300, MPI_COMM_WORLD, MPI_INFO_NULL, &req);
    } else {
        MPI_Precv_init(buf, total_parts, (MPI_Count)part_size, MPI_BYTE, 0, 300, MPI_COMM_WORLD, MPI_INFO_NULL, &req);
    }

    double delay_time_us = gamma_us_per_mb * ((double)part_size / (1024.0 * 1024.0));
    double total_comm_time = 0.0;

    #pragma omp parallel num_threads(num_threads) shared(req, total_comm_time)
    {
        int tid = omp_get_thread_num();
        for (int iter = -WARMUP; iter < ITERS; iter++) {
            #pragma omp barrier
            #pragma omp master
            MPI_Barrier(MPI_COMM_WORLD);
            #pragma omp barrier

            double t_start = MPI_Wtime();

            #pragma omp master
            MPI_Start(&req);
            #pragma omp barrier

            if (rank == 0) {
                if (tid == num_threads - 1) {
                    delay_us(delay_time_us);
                }
                for (int p = 0; p < theta; p++) {
                    MPI_Pready(tid * theta + p, req);
                }
            }

            #pragma omp barrier
            #pragma omp master
            MPI_Wait(&req, MPI_STATUS_IGNORE);

            double t_end = MPI_Wtime();
            if (iter >= 0 && tid == 0) {
                double comm_only = (t_end - t_start) - (delay_time_us * 1e-6);
                if (comm_only < 0.0) comm_only = t_end - t_start;
                total_comm_time += comm_only;
            }
        }
    }

    MPI_Request_free(&req);
    free(buf);
    return (total_comm_time / ITERS) * 1e6;
}

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int num_threads = 4;

    FILE *csv = NULL;
    if (rank == 0) {
        csv = fopen("results.csv", "w");
        fprintf(csv, "exp,msg_size,single,many,part,ratio\n");

        printf("=====================================================================================\n");
        printf("EXP 1: Small Message Contention (32 Partitions: 4 Threads x 8 Parts, gamma = 0)\n");
        printf("Replicating Fig 5 & 7 | Showing Communication Time (us) -> Lower is Better\n");
        printf("=====================================================================================\n");
        printf("%-12s | %-16s | %-16s | %-16s | %-12s\n", 
               "Msg Size(B)", "Pt2Pt_Single(us)", "Pt2Pt_Many(us)", "Pt2Pt_Part(us)", "Many/Single");
        printf("-------------------------------------------------------------------------------------\n");
    }

    // EXP 1: theta = 8 (32 partitions total, matching paper's 32-partition contention test)
    int theta_small = 8;
    size_t small_sizes[] = {256, 1024, 4096, 16384, 65536, 262144};
    for (int i = 0; i < 6; i++) {
        size_t msg_size = small_sizes[i];
        size_t part_size = msg_size / (num_threads * theta_small);

        double t_single = bench_pt2pt_single(rank, num_threads, theta_small, part_size, 0.0);
        double t_many   = bench_pt2pt_many(rank, num_threads, theta_small, part_size, 0.0);
        double t_part   = bench_pt2pt_part(rank, num_threads, theta_small, part_size, 0.0);

        if (rank == 0) {
            double penalty = t_many / t_single;
            printf("%-12zu | %-16.2f | %-16.2f | %-16.2f | x%-11.2f\n",
                   msg_size, t_single, t_many, t_part, penalty);
            fprintf(csv, "exp1,%zu,%.2f,%.2f,%.2f,%.2f\n", msg_size, t_single, t_many, t_part, penalty);
        }
    }

    if (rank == 0) {
        printf("\n=====================================================================================\n");
        printf("EXP 2: Large Message Early-Bird Effect (4 Partitions: 4 Threads x 1 Part, gamma = 30)\n");
        printf("Replicating Fig 8 | Showing Perceived Bandwidth (GB/s) -> Higher is Better\n");
        printf("=====================================================================================\n");
        printf("%-12s | %-16s | %-16s | %-16s | %-12s\n", 
               "Msg Size(B)", "Single BW(GB/s)", "Many BW(GB/s)", "Part BW(GB/s)", "EarlyBird Gain");
        printf("-------------------------------------------------------------------------------------\n");
    }

    // EXP 2: theta = 1 (4 partitions total, matching Fig 8's 4-thread test)
    // gamma = 30 us/MB scaled for Apple M4 memory speed to show the crossover around 64KB-256KB
    int theta_large = 1;
    double gamma = 30.0;
    size_t large_sizes[] = {4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 33554432};
    for (int i = 0; i < 8; i++) {
        size_t msg_size = large_sizes[i];
        size_t part_size = msg_size / (num_threads * theta_large);

        double t_single = bench_pt2pt_single(rank, num_threads, theta_large, part_size, gamma);
        double t_many   = bench_pt2pt_many(rank, num_threads, theta_large, part_size, gamma);
        double t_part   = bench_pt2pt_part(rank, num_threads, theta_large, part_size, gamma);

        if (rank == 0) {
            double gb = (double)msg_size / 1e9;
            double bw_single = gb / (t_single * 1e-6);
            double bw_many   = gb / (t_many * 1e-6);
            double bw_part   = gb / (t_part * 1e-6);
            double gain      = bw_many / bw_single;
            printf("%-12zu | %-16.2f | %-16.2f | %-16.2f | x%-11.2f\n",
                   msg_size, bw_single, bw_many, bw_part, gain);
            fprintf(csv, "exp2,%zu,%.2f,%.2f,%.2f,%.2f\n", msg_size, bw_single, bw_many, bw_part, gain);
        }
    }

    if (rank == 0 && csv) fclose(csv);
    MPI_Finalize();
    return 0;
}