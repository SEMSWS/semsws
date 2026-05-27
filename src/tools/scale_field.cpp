/**
 * @file scale_field.cpp
 * @brief Scale an ADIOS2 .bp MaterialField by a scalar factor
 *
 * Reads a field from a .bp file, multiplies all values by the given scalar,
 * and writes the result to an output .bp file.
 *
 * Usage:
 *   mpirun -np N scale_field --input A.bp --output B.bp --scale 1.5e-4 \
 *       --device cuda
 */

#include "io/ADIOS2IO.hpp"
#include "material/MaterialField.hpp"
#include <mfem.hpp>
#include <iostream>
#include <string>
#include <cstdlib>

using namespace SEM;
using namespace mfem;

int main(int argc, char* argv[]) {
    Mpi::Init(argc, argv);
    Hypre::Init();

    std::string input_file, output_file;
    std::string device_str = "cpu";
    real_t scale = 1.0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) input_file = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_file = argv[++i];
        else if (arg == "--scale" && i + 1 < argc) scale = std::atof(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) device_str = argv[++i];
    }

    if (input_file.empty() || output_file.empty()) {
        if (Mpi::Root()) {
            std::cerr << "Usage: scale_field --input A.bp --output B.bp "
                      << "--scale <value> [--device cpu|cuda]\n";
        }
        return 1;
    }

    Device device(device_str);

    MPI_Comm comm = MPI_COMM_WORLD;

    MaterialField field = LoadFieldBP(input_file, "data", comm);
    field.Data() *= scale;

    SaveFieldBP(output_file, "data", field, input_file, comm);

    return 0;
}
