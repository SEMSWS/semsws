/**
 * @file sum_kernels.cpp
 * @brief Sum multiple per-source kernels into a total kernel
 *
 * Reads kernels from ADIOS2 .bp files and sums them element-wise.
 * Mesh and order are read from config.yaml for consistency.
 *
 * Usage:
 *   mpirun -np N sum_kernels \
 *       --config sim.yaml \
 *       --inputs kernel_src000.bp,kernel_src001.bp,kernel_src002.bp \
 *       --output kernel_total.bp \
 *       --device cuda
 */

#include "io/ADIOS2IO.hpp"
#include "material/MaterialField.hpp"
#include <mfem.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>

using namespace SEM;
using namespace mfem;

static std::vector<std::string> SplitComma(const std::string& s) {
    std::vector<std::string> result;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        if (!token.empty()) {
            result.push_back(token);
        }
    }
    return result;
}

int main(int argc, char* argv[]) {
    Mpi::Init(argc, argv);
    Hypre::Init();

    std::string config_file, inputs_str, output_file;
    std::string device_str = "cpu";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) config_file = argv[++i];
        else if (arg == "--inputs" && i + 1 < argc) inputs_str = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_file = argv[++i];
        else if (arg == "--device" && i + 1 < argc) device_str = argv[++i];
    }

    if (config_file.empty() || inputs_str.empty() || output_file.empty()) {
        if (Mpi::Root()) {
            std::cerr << "Usage: sum_kernels --config C.yaml "
                      << "--inputs a.bp,b.bp,... --output total.bp "
                      << "[--device cpu|cuda]\n";
        }
        return 1;
    }

    Device device(device_str);

    std::vector<std::string> input_files = SplitComma(inputs_str);
    if (input_files.empty()) {
        if (Mpi::Root()) {
            std::cerr << "Error: No input files specified\n";
        }
        return 1;
    }

    MPI_Comm comm = MPI_COMM_WORLD;

    // Load first kernel
    if (Mpi::Root()) {
        std::cout << "Loading kernel [0]: " << input_files[0] << std::endl;
    }
    MaterialField total = LoadFieldBP(input_files[0], "data", comm);

    // Sum remaining kernels
    for (size_t i = 1; i < input_files.size(); i++) {
        if (Mpi::Root()) {
            std::cout << "Loading kernel [" << i << "]: " << input_files[i] << std::endl;
        }
        MaterialField k = LoadFieldBP(input_files[i], "data", comm);
        total.Data() += k.Data();
    }

    // Save total kernel
    if (Mpi::Root()) {
        std::cout << "Saving total kernel (" << input_files.size()
                  << " sources): " << output_file << std::endl;
    }
    SaveFieldBP(output_file, "data", total, config_file, comm);

    if (Mpi::Root()) {
        std::cout << "Done." << std::endl;
    }

    return 0;
}
