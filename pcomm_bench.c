/*
 * pcomm_bench.c -- small-scale replication of Gillis et al. (ICPP'23)
 * "Quantifying the Performance Benefits of Partitioned Communication in MPI"
 *
 * Variants:
 *   single         : thread barrier, then ONE persistent send  (bulk sync, Fig. 1)
 *   many           : one persistent send per partition, per-thread dup'd comm (Fig. 2)
 *   part           : MPI 4.0 partitioned send, Pready only (naive)
 *   part_progress  : same, but MPI_Test polls to drive progress while waiting
 *                    for completion (tests the "no progress" hypothesis)
 *
 * Time-to-solution = [MPI_Start ... receiver completion] - artificial delay.
 * Receiver completion is captured with an MPI_Barrier after MPI_Wait on both ranks.
 *
 * Build:  mpicc -O2 -Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include \
 *               pcomm_bench.c -o pcomm_bench -L$(brew --prefix libomp)/lib -lomp -lm
 * Run:    mpirun --bind-to none -np 2 ./pcomm_bench [threads=4] [theta_exp1=8]
 * Tip:    export OMP_WAIT_POLICY=active   (spin, avoids sleep/wake noise)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sched.h>
#include <mpi.h>
#include <omp.h>

#define WARMUP     15
#define ITERS      100
#define MAX_RETRY  5          /* paper uses 50; keep small on a laptop */
#define CI_TOL     0.05       /* rerun if CI half-width > 5% of |mean| */
#define MB         1e6        /* ONE convention everywhere: decimal MB / GB */

typedef enum { V_SINGLE, V_MANY, V_PART, V_PART_PROG, V_COUNT } variant_t;
static const char *vname[V_COUNT] = {"single", "many", "part", "part_progress"};

typedef struct { double mean, ci, median; int neg, attempts; } Stats;

static void delay_us(double us) {
    if (us <= 0.0) return;
    double s = MPI_Wtime();
    while ((MPI_Wtime() - s) * 1e6 < us) {}
}

/* ---------- statistics ---------- */
static double t90(int df) {   /* two-sided 90% Student-t critical value (Cornish-Fisher) */
    double z = 1.6448536, z3 = z*z*z, z5 = z3*z*z;
    return z + (z3 + z) / (4.0*df) + (5*z5 + 16*z3 + 3*z) / (96.0*df*df);
}
static int cmp_d(const void *a, const void *b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}
static void compute_stats(const double *s, int n, Stats *st) {
    double sum = 0, sq = 0; st->neg = 0;
    for (int i = 0; i < n; i++) { sum += s[i]; if (s[i] < 0) st->neg++; }
    st->mean = sum / n;
    for (int i = 0; i < n; i++) sq += (s[i]-st->mean)*(s[i]-st->mean);
    double sd = sqrt(sq / (n-1));
    st->ci = t90(n-1) * sd / sqrt((double)n);
    double *c = malloc(n*sizeof(double)); memcpy(c, s, n*sizeof(double));
    qsort(c, n, sizeof(double), cmp_d);
    st->median = (n % 2) ? c[n/2] : 0.5*(c[n/2-1] + c[n/2]);
    free(c);
}

/* ---------- one benchmark run: fills samples[ITERS] (microseconds) ---------- */
static void bench(variant_t v, int rank, int nth, int theta, size_t part_size,
                  double gamma, double *samples)
{
    int total = nth * theta;
    size_t total_size = part_size * total;
    char *buf = calloc(total_size, 1);
    MPI_Request req = MPI_REQUEST_NULL, *reqs = NULL;
    MPI_Comm *comms = NULL;
    double dl = gamma * ((double)part_size / MB);       /* D = gamma * S_part  [us] */

    if (v == V_SINGLE) {
        if (rank == 0) MPI_Send_init(buf, (int)total_size, MPI_BYTE, 1, 100, MPI_COMM_WORLD, &req);
        else           MPI_Recv_init(buf, (int)total_size, MPI_BYTE, 0, 100, MPI_COMM_WORLD, &req);
    } else if (v == V_MANY) {
        comms = malloc(nth * sizeof(MPI_Comm));
        reqs  = malloc(total * sizeof(MPI_Request));
        for (int t = 0; t < nth; t++) {
            MPI_Comm_dup(MPI_COMM_WORLD, &comms[t]);
            for (int p = 0; p < theta; p++) {
                int idx = t*theta + p;
                if (rank == 0) MPI_Send_init(buf + idx*part_size, (int)part_size, MPI_BYTE, 1, 200+idx, comms[t], &reqs[idx]);
                else           MPI_Recv_init(buf + idx*part_size, (int)part_size, MPI_BYTE, 0, 200+idx, comms[t], &reqs[idx]);
            }
        }
    } else {
        int part_tag = (v == V_PART) ? 300 : 301;
        if (rank == 0) MPI_Psend_init(buf, total, (MPI_Count)part_size, MPI_BYTE, 1, part_tag, MPI_COMM_WORLD, MPI_INFO_NULL, &req);
        else           MPI_Precv_init(buf, total, (MPI_Count)part_size, MPI_BYTE, 0, part_tag, MPI_COMM_WORLD, MPI_INFO_NULL, &req);
    }

    #pragma omp parallel num_threads(nth) shared(req, reqs, samples)
    {
        int tid = omp_get_thread_num();
        for (int iter = -WARMUP; iter < ITERS; iter++) {
            #pragma omp barrier
            #pragma omp master
            MPI_Barrier(MPI_COMM_WORLD);
            #pragma omp barrier

            double t0 = MPI_Wtime();

            if (v == V_SINGLE) {
                if (rank == 0 && tid == nth-1) delay_us(dl);
                #pragma omp barrier
                #pragma omp master
                { MPI_Start(&req); MPI_Wait(&req, MPI_STATUS_IGNORE); }

            } else if (v == V_MANY) {
                if (rank == 0 && tid == nth-1) delay_us(dl);
                for (int p = 0; p < theta; p++) MPI_Start(&reqs[tid*theta + p]);
                MPI_Waitall(theta, &reqs[tid*theta], MPI_STATUSES_IGNORE);

            } else { /* V_PART or V_PART_PROG */
                #pragma omp master
                MPI_Start(&req);
                #pragma omp barrier
                if (rank == 0) {
                    if (tid == nth-1) delay_us(dl);
                    for (int p = 0; p < theta; p++) {
                        MPI_Pready(tid*theta + p, req);
                    }
                }
                #pragma omp barrier
                #pragma omp master
                {
                    if (v == V_PART) {
                        MPI_Wait(&req, MPI_STATUS_IGNORE);
                    } else {
                        int complete = 0;
                        double deadline = MPI_Wtime() + 30.0;
                        while (!complete && MPI_Wtime() < deadline) {
                            MPI_Test(&req, &complete, MPI_STATUS_IGNORE);
                            if (!complete) sched_yield();
                        }
                        if (!complete) {
                            fprintf(stderr, "rank %d: part_progress timed out in iteration %d\n",
                                    rank, iter);
                            MPI_Abort(MPI_COMM_WORLD, 2);
                        }
                    }
                }
            }

            /* end-of-iteration sync: captures receiver completion (same cost for all variants) */
            #pragma omp barrier
            #pragma omp master
            MPI_Barrier(MPI_COMM_WORLD);
            #pragma omp barrier
            double t1 = MPI_Wtime();

            if (iter >= 0 && tid == 0)
                samples[iter] = ((t1 - t0) * 1e6) - dl;   /* subtract artificial delay, no clamping */
        }
    }

    if (v == V_MANY) {
        for (int i = 0; i < total; i++) MPI_Request_free(&reqs[i]);
        for (int t = 0; t < nth; t++) MPI_Comm_free(&comms[t]);
        free(comms); free(reqs);
    } else {
        MPI_Request_free(&req);
    }
    free(buf);
}

/* run with CI-based retry; the retry decision is broadcast so both ranks stay in lockstep */
static void run_point(variant_t v, int rank, int nth, int theta, size_t part_size,
                      double gamma, Stats *st)
{
    if (rank == 0)
        fprintf(stderr, "running %s: size_per_part=%zu gamma=%.1f\n",
                vname[v], part_size, gamma);
    double samples[ITERS];
    int attempt;
    for (attempt = 1; attempt <= MAX_RETRY; attempt++) {
        bench(v, rank, nth, theta, part_size, gamma, samples);
        int again = 0;
        if (rank == 0) {
            compute_stats(samples, ITERS, st);
            again = (st->ci > CI_TOL * fabs(st->mean));
        }
        MPI_Bcast(&again, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (!again) break;
    }
    st->attempts = (attempt > MAX_RETRY) ? MAX_RETRY : attempt;
}

/* one-way bandwidth estimate (MB/s) from a ping-pong, used for the model overlay */
static double measure_beta(int rank) {
    size_t n = 32u << 20;
    char *b = calloc(n, 1);
    double t = 0;
    for (int i = -3; i < 20; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double s = MPI_Wtime();
        if (rank == 0) { MPI_Send(b, (int)n, MPI_BYTE, 1, 1, MPI_COMM_WORLD);
                         MPI_Recv(b, (int)n, MPI_BYTE, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE); }
        else           { MPI_Recv(b, (int)n, MPI_BYTE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                         MPI_Send(b, (int)n, MPI_BYTE, 0, 2, MPI_COMM_WORLD); }
        if (i >= 0) t += MPI_Wtime() - s;
    }
    free(b);
    double beta = 0;
    if (rank == 0) beta = ((double)n / MB) / ((t / 20.0) / 2.0);   /* MB/s */
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return beta;
}

int main(int argc, char **argv) {
    int provided, rank, size;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (provided < MPI_THREAD_MULTIPLE || size != 2) {
        if (rank == 0) fprintf(stderr, "Need MPI_THREAD_MULTIPLE and exactly 2 ranks (got provided=%d, size=%d)\n", provided, size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int nth      = (argc > 1) ? atoi(argv[1]) : 4;
    int theta_s  = (argc > 2) ? atoi(argv[2]) : 8;

    double beta = measure_beta(rank);   /* MB/s */
    FILE *csv = NULL;
    if (rank == 0) {
        csv = fopen("results.csv", "w");
        fprintf(csv, "exp,gamma_us_per_MB,msg_size_B,method,mean_us,ci90_us,median_us,perceived_bw_GBps,eta_vs_single,eta_model,attempts,neg_samples\n");
        printf("threads=%d  theta(exp1)=%d  measured beta = %.0f MB/s (%.1f GB/s)\n", nth, theta_s, beta, beta/1e3);
        printf("Note: ci90 = 90%% Student-t half-width over %d iters; attempts>1 means CI exceeded %.0f%% and was rerun.\n\n", ITERS, CI_TOL*100);
    }

    /* ---------------- EXP 1: small messages, gamma = 0 ---------------- */
    size_t small_sizes[] = {64*8, 1024, 4096, 16384, 65536, 262144, 1048576};
    int nsmall = sizeof(small_sizes)/sizeof(small_sizes[0]);
    if (rank == 0) printf("EXP 1: %d threads x %d partitions, gamma=0  (time in us, lower is better)\n"
                          "%-10s", nth, theta_s, "size(B)");
    if (rank == 0) { for (int v = 0; v < V_COUNT; v++) printf(" | %-22s", vname[v]); printf("\n"); }
    for (int i = 0; i < nsmall; i++) {
        size_t msg = small_sizes[i], psz = msg / (nth*theta_s);
        if (psz == 0) continue;
        Stats st[V_COUNT];
        for (int v = 0; v < V_COUNT; v++) run_point(v, rank, nth, theta_s, psz, 0.0, &st[v]);
        if (rank == 0) {
            printf("%-10zu", msg);
            for (int v = 0; v < V_COUNT; v++) {
                printf(" | %8.2f +/-%6.2f x%-3d", st[v].mean, st[v].ci, st[v].attempts);
                fprintf(csv, "exp1,0,%zu,%s,%.3f,%.3f,%.3f,%.4f,%.3f,,%d,%d\n", msg, vname[v],
                        st[v].mean, st[v].ci, st[v].median,
                        st[v].mean > 0 ? (double)msg/1e9/(st[v].mean*1e-6) : NAN,
                        st[V_SINGLE].mean / st[v].mean, st[v].attempts, st[v].neg);
            }
            printf("\n"); fflush(stdout);
        }
    }

    /* ---------------- EXP 2: large messages, gamma sweep ---------------- */
    double gammas[] = {0.0, 10.0, 30.0, 100.0};
    size_t large_sizes[] = {65536, 262144, 1048576, 4194304, 8388608, 16777216, 33554432};
    int ng = sizeof(gammas)/sizeof(gammas[0]), nl = sizeof(large_sizes)/sizeof(large_sizes[0]);
    for (int g = 0; g < ng; g++) {
        double gamma = gammas[g];
        double model = (double)nth / fmax((double)nth - gamma*1e-6*beta, 1.0);   /* Eq. 4, theta=1 */
        if (rank == 0) printf("\nEXP 2: %d threads x 1 partition, gamma=%.0f us/MB   (model eta = %.3f)\n"
                              "%-10s", nth, gamma, model, "size(B)");
        if (rank == 0) { for (int v = 0; v < V_COUNT; v++) printf(" | %-24s", vname[v]); printf("\n"); }
        for (int i = 0; i < nl; i++) {
            size_t msg = large_sizes[i], psz = msg / nth;
            Stats st[V_COUNT];
            for (int v = 0; v < V_COUNT; v++) run_point(v, rank, nth, 1, psz, gamma, &st[v]);
            if (rank == 0) {
                printf("%-10zu", msg);
                for (int v = 0; v < V_COUNT; v++) {
                    double bw = st[v].mean > 0 ? (double)msg/1e9/(st[v].mean*1e-6) : NAN;
                    printf(" | %6.2f GB/s eta=%-5.2f x%-2d", bw, st[V_SINGLE].mean / st[v].mean, st[v].attempts);
                    fprintf(csv, "exp2,%.0f,%zu,%s,%.3f,%.3f,%.3f,%.4f,%.3f,%.3f,%d,%d\n", gamma, msg, vname[v],
                            st[v].mean, st[v].ci, st[v].median, bw,
                            st[V_SINGLE].mean / st[v].mean, model, st[v].attempts, st[v].neg);
                }
                printf("\n"); fflush(stdout);
            }
        }
    }

    if (csv) fclose(csv);
    MPI_Finalize();
    return 0;
}