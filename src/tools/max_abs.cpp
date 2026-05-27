/**
 * @file max_abs.cpp
 * @brief Compute the global max absolute value of an ADIOS2 .bp MaterialField
 *
 * Reads a field from a .bp file and computes max(|data|) across all MPI ranks.
 * Result is written to an output file.
 *
 * Usage:
 *   mpirun -np N max_abs --field A.bp --output result.txt --device cuda
 *
 * Output file contains a single line:
 *   MAX_ABS=<value>
 */

#include "io/ADIOS2IO.hpp"
#include "material/MaterialField.hpp"
#include <mfem.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cmath>

using namespace SEM;
using namespace mfem;

int main(int argc, char* argv[]) {
    Mpi::Init(argc, argv);
    Hypre::Init();

    std::string field_file, output;
    std::string device_str = "cpu";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--field" && i + 1 < argc) field_file = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output = argv[++i];
        else if (arg == "--device" && i + 1 < argc) device_str = argv[++i];
    }

    if (field_file.empty() || output.empty()) {
        if (Mpi::Root()) {
            std::cerr << "Usage: max_abs --field A.bp --output result.txt "
                      << "[--device cpu|cuda]\n";
        }
        return 1;
    }

    Device device(device_str);

    MPI_Comm comm = MPI_COMM_WORLD;

    MaterialField field = LoadFieldBP(field_file, "data", comm);
    const Vector& data = field.Data();
    const int n = data.Size();

    // Host-side max reduction
    const real_t* d = data.HostRead();
    real_t local_max = 0.0;
    for (int i = 0; i < n; i++) {
        real_t a = std::abs(d[i]);
        if (a > local_max) local_max = a;
    }

    real_t global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MFEM_MPI_REAL_T, MPI_MAX, comm);

    if (Mpi::Root()) {
        std::ofstream ofs(output);
        if (!ofs) {
            std::cerr << "Error: cannot open output file: " << output << "\n";
            return 1;
        }
        ofs << std::setprecision(17) << "MAX_ABS=" << global_max << std::endl;
    }

    return 0;
}
