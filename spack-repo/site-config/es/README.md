# Earth Simulator (JAMSTEC) site config for SEMSWS

Spack build settings for both the GPU partition (NVIDIA A100, sm_80) and the
CPU partition. Same shape as the LUMI config (load.sh + packages.yaml), but
ES has two targets so both flavours ship side by side.

```text
load_gpu.sh           module load + Spack environment (NVHPC + CUDA)
load_cpu.sh           module load + Spack environment (gcc + system OpenMPI)
packages_gpu.yaml     externals: nvhpc / hpcx-ompi / cuda 12.0 / cmake
                      Spack-built (%gcc): hdf5 / metis / netcdf / openblas
packages_cpu.yaml     externals: system OpenMPI 4.0.5 / cmake
                      Spack-built (%gcc): hdf5 / metis / netcdf / openblas
```

## First-time setup

```bash
# 0) Working directory
export WORK=/S/data01/G3506/${USER}/program_test   # change to your path
mkdir -p "$WORK"

# 1) Clone Spack (one time; can be skipped if the site share already has it)
git clone --depth=1 https://github.com/spack/spack.git "${WORK}/spack"

# 2) Clone SEMSWS
cd "$WORK"
git clone https://github.com/SEMSWS/semsws.git
cd SEMSWS

# 3) Drop loader scripts into place
cp spack-repo/site-config/es/load_gpu.sh ~/load_gpu.sh
cp spack-repo/site-config/es/load_cpu.sh ~/load_cpu.sh
chmod +x ~/load_gpu.sh ~/load_cpu.sh
```

From here, choose **(A) single target** or **(B) GPU + CPU side by side**.

## (A) Single target (LUMI-style)

For "GPU only" or "CPU only" use cases. Same flow as the LUMI README.

```bash
# GPU example
source ~/load_gpu.sh
mkdir -p ~/.spack
[ -f ~/.spack/packages.yaml ] && mv ~/.spack/packages.yaml ~/.spack/packages.yaml.bak
cp spack-repo/site-config/es/packages_gpu.yaml ~/.spack/packages.yaml
# Or via SPACK_USER_CONFIG_PATH (load_gpu.sh points it at $WORK/.spack):
cp spack-repo/site-config/es/packages_gpu.yaml "$SPACK_USER_CONFIG_PATH/packages.yaml"

spack compiler find                    # registers nvhpc@24.7 + gcc@8.5
spack repo add "$WORK/SEMSWS/spack-repo"
spack install semsws@main +cuda cuda_arch=80 precision=single +gpu_aware_mpi \
  cxxflags=="-noswitcherror" cflags=="-noswitcherror" ldlibs=="-lstdc++fs" \
  ^hypre@2.33 %nvhpc
```

For CPU, swap to `load_cpu.sh` + `packages_cpu.yaml` and run
`spack install semsws@main precision=single %gcc`.

## (B) GPU and CPU side by side (Spack environment)

To install both and switch between them, isolate via Spack environments
rather than swapping `SPACK_USER_CONFIG_PATH` — env-scoped configuration
is harder to break by accident.

### GPU env

```bash
source ~/load_gpu.sh
spack compiler find
spack env create semsws-gpu
spack env activate semsws-gpu
spack config edit                       # opens an editor
# Paste the GPU spack.yaml below (packages_gpu.yaml content nested under packages:)
```

`semsws-gpu` `spack.yaml`:

```yaml
spack:
  repos:
  - /S/data01/G3506/<USER>/program_test/SEMSWS/spack-repo

  packages:
    cmake:
      externals:
      - spec: cmake@3.26.5
        prefix: /usr
      buildable: false
    nvhpc:
      externals:
      - spec: nvhpc@24.7 ~mpi
        prefix: /opt/share/NVIDIAHPCSDK/24.7
        extra_attributes:
          compilers:
            c:       /opt/share/NVIDIAHPCSDK/24.7/Linux_x86_64/24.7/compilers/bin/nvc
            cxx:     /opt/share/NVIDIAHPCSDK/24.7/Linux_x86_64/24.7/compilers/bin/nvc++
            fortran: /opt/share/NVIDIAHPCSDK/24.7/Linux_x86_64/24.7/compilers/bin/nvfortran
      buildable: false
    openmpi:
      externals:
      - spec: openmpi@4.1.7 +cuda fabrics=ucx
        prefix: /opt/share/NVIDIAHPCSDK/24.7/Linux_x86_64/24.7/comm_libs/12.5/hpcx/hpcx-2.19/ompi
      buildable: false
    cuda:
      externals:
      - spec: cuda@12.0.0
        prefix: /opt/share/CUDA/12.0.0
      buildable: false
    hdf5:
      variants: +mpi+hl
      require: '%gcc'
    metis:
      require: '%gcc'
    # ADIOS2 2.12+ needs -lstdc++fs on GCC < 9; ES uses GCC 8.5 → pin <=2.11.
    adios2:
      require:
      - '%gcc'
      - '@:2.11'
    netcdf-c:       { require: '%gcc' }
    netcdf-cxx4:    { require: '%gcc' }
    netcdf-fortran: { require: '%gcc' }
    blas:   { require: openblas }
    lapack: { require: openblas }
    openblas:
      require: '%gcc'
    hypre:
      variants: +shared
    all:
      providers:
        mpi: [openmpi]
      variants: cuda_arch=80

  specs:
  - semsws@main +cuda cuda_arch=80 precision=single +gpu_aware_mpi cxxflags=="-noswitcherror" cflags=="-noswitcherror" ldlibs=="-lstdc++fs" ^hypre@2.33 %nvhpc

  concretizer:
    unify: true
  config:
    build_jobs: 4
```

```bash
spack concretize -f
spack install --fail-fast
spack env deactivate
```

### CPU env

```bash
source ~/load_cpu.sh
spack compiler find
spack env create semsws-cpu
spack env activate semsws-cpu
spack config edit
# Paste the CPU spack.yaml below
```

`semsws-cpu` `spack.yaml`:

```yaml
spack:
  repos:
  - /S/data01/G3506/<USER>/program_test/SEMSWS/spack-repo

  packages:
    cmake:
      externals:
      - spec: cmake@3.26.5
        prefix: /usr
      buildable: false
    openmpi:
      externals:
      - spec: openmpi@4.0.5
        prefix: /opt/share/OpenMPI/4.0.5
      buildable: false
    hdf5:
      variants: +mpi+hl
      require: '%gcc'
    metis:
      require: '%gcc'
    # ADIOS2 2.12+ needs -lstdc++fs on GCC < 9; ES uses GCC 8.5 → pin <=2.11.
    adios2:
      require:
      - '%gcc'
      - '@:2.11'
    netcdf-c:       { require: '%gcc' }
    netcdf-cxx4:    { require: '%gcc' }
    netcdf-fortran: { require: '%gcc' }
    blas:   { require: openblas }
    lapack: { require: openblas }
    openblas:
      require: '%gcc'
    hypre:
      variants: +shared
    all:
      providers:
        mpi: [openmpi]

  specs:
  - semsws@main precision=single %gcc

  concretizer:
    unify: true
  config:
    build_jobs: 4
```

```bash
spack concretize -f
spack install --fail-fast
spack env deactivate
```

## Subsequent sessions

```bash
# GPU
source ~/load_gpu.sh
spack env activate semsws-gpu          # only for the (B) flow
spack load semsws

# CPU
source ~/load_cpu.sh
spack env activate semsws-cpu
spack load semsws
```

## Notes

- **Module conflicts**: `NVIDIAHPCSDK/24.7/...` and `OpenMPI/4.0.5` cannot be
  loaded together. Both `load_gpu.sh` and `load_cpu.sh` start with
  `module purge`. Always re-source the right `~/load_xxx.sh` when switching.
- **Why CUDA 12.0**: the CUDA 12.5 bundled with NVHPC SDK 24.7 uses a split
  layout that hides `cusparse.h` from MFEM. The standalone
  `/opt/share/CUDA/12.0.0` flat-layout module is used instead.
- **hypre 2.33 is pinned**: hypre 3.x trips two issues with ES gcc 8.5 +
  nvcc — a `cstddef` `extern "C"` clash and a CMake `FindMPI` mismatch with
  NVHPC. The shadow recipe under `spack-repo/packages/hypre/` already
  disables cuSOLVER and works around the FSAI device bug.
