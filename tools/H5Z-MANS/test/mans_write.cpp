#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <hdf5.h>
#include <mpi.h>
#include "mans_defs.h"

#define FILTER_ID_MANS 32001
#define CHECK_H5(x) do { if ((x) < 0) { std::fprintf(stderr, "HDF5 failed: %s\n", #x); std::exit(1); } } while (0)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    char in[64], out[64];
    std::snprintf(in, sizeof(in), "rank_%04d.bin", rank);
    std::snprintf(out, sizeof(out), "rank_%04d.h5", rank);
    std::FILE* fp = std::fopen(in, "rb");
    if (!fp) { std::fprintf(stderr, "rank %d missing input: %s\n", rank, in); return 1; }
    std::fseek(fp, 0, SEEK_END);
    long bytes_l = std::ftell(fp);
    std::rewind(fp);
    if (bytes_l <= 0 || (bytes_l % (long)sizeof(uint32_t)) != 0) { std::fprintf(stderr, "bad input size: %s\n", in); return 1; }
    size_t bytes = (size_t)bytes_l, elems = bytes / sizeof(uint32_t);
    std::vector<uint32_t> data(elems);
    if (std::fread(data.data(), 1, bytes, fp) != bytes) { std::fprintf(stderr, "read failed: %s\n", in); return 1; }
    std::fclose(fp);

    mans::MansParams p{};
    p.backend = mans::Backend::CPU; p.dtype = mans::DataType::U32; p.adm_threshold = 4000;
    p.adm_decide_threads = 16; p.adm_center_calc_threads = 32; p.adm_encode_threads = 32;
    p.adm_warp_reduce_threads = 32; p.adm_fill_tail_threads = 16; p.adm_write_back_threads = 16;
    p.adm_restore_signals_threads = 32; p.adm_decode_values_threads = 16;
    std::vector<unsigned int> cd(sizeof(p) / sizeof(unsigned int), 0);
    std::memcpy(cd.data(), &p, sizeof(p));

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    hid_t file = H5Fcreate(out, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    hsize_t dims[1] = { (hsize_t)elems }, chunk[1] = { (hsize_t)((1 << 20) / sizeof(uint32_t)) };
    if (chunk[0] == 0 || chunk[0] > dims[0]) chunk[0] = dims[0];
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    CHECK_H5(H5Pset_chunk(dcpl, 1, chunk));
    CHECK_H5(H5Pset_filter(dcpl, FILTER_ID_MANS, 0, cd.size(), cd.data()));
    hid_t dset = H5Dcreate2(file, "data", H5T_NATIVE_UINT, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    CHECK_H5(H5Dwrite(dset, H5T_NATIVE_UINT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()));
    CHECK_H5(H5Fflush(file, H5F_SCOPE_GLOBAL));
    double t1 = MPI_Wtime();

    H5Dclose(dset); H5Sclose(space); H5Fclose(file);
    double sec = t1 - t0, max_sec = sec, sum_bytes = (double)bytes;
    MPI_Reduce(&sec, &max_sec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_bytes, &sum_bytes, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        double mib = sum_bytes / (1024.0 * 1024.0), bw = (max_sec > 0.0) ? (mib / max_sec) : 0.0;
        std::printf("mans_write ranks=%d total=%.1f MiB time=%.4f s throughput=%.2f MiB/s\n", nprocs, mib, max_sec, bw);
    }
    MPI_Finalize();
    return 0;
}
