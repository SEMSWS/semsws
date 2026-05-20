/**
 * @file SimulationFacadeSetup.cpp
 * @brief Setup methods for SimulationFacade class
 *
 * Contains all configuration and setup methods:
 * - LoadConfig
 * - SetupMeshFromConfig
 * - SetupMaterialFromConfig
 * - SetupOperatorFromConfig
 * - SetupSourcesFromConfig
 * - SetupReceiversFromConfig
 * - SetupFromConfig
 */

#include "simulation/SimulationFacade.hpp"
#include "operator/OperatorFactory.hpp"
#include "common/Types.hpp"
#include "config/ConfigLoaders.hpp"
#include "srcrecv/SrcRecvLoaders.hpp"
#include "config/ConfigTypes.hpp"
#include "material/MaterialFactory.hpp"
#include "material/isotropic_elastic/IsotropicElasticMaterial.hpp"
#include "material/isotropic_acoustic/IsotropicAcousticMaterial.hpp"
#include "material/ElasticMaterialBase.hpp"
#include "material/AcousticMaterialBase.hpp"
#include "operator/WaveOperatorBase2D.hpp"
#include "operator/WaveOperatorBase3D.hpp"
#include "util/Profiler.hpp"
#include "material/isotropic_acoustic/IsotropicAcousticIO.hpp"
#include "material/isotropic_elastic/IsotropicElasticIO.hpp"

#include <cerrno>
#ifndef __linux__
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <sys/stat.h>
#endif

namespace SEM {

// =============================================================================
// Helper Functions
// =============================================================================

namespace {

/// Create directory (cross-platform)
bool CreateDirectory(const std::string& path) {
#ifdef __linux__
    if (mkdir(path.c_str(), 0755) == 0 || errno == EEXIST) {
#else
    if (_mkdir(path.c_str()) == 0 || errno == EEXIST) {
#endif
        return true;
    }
    return false;
}

/// Validate local DOFs fit in int (for gather_map arrays)
void ValidateLocalDOFSize(const ParFiniteElementSpace& fes, MPI_Comm comm, int rank) {
    HYPRE_BigInt local_tdofs = fes.GetTrueVSize();
    HYPRE_BigInt global_tdofs = fes.GlobalTrueVSize();

    constexpr HYPRE_BigInt INT_MAX_VAL = 2147483647LL;

    // Check local DOFs fit in int (required for gather_map arrays)
    if (local_tdofs > INT_MAX_VAL) {
        if (rank == 0) {
            std::cerr << "\n[ERROR] Local DOFs (" << local_tdofs
                      << ") exceed 32-bit integer limit (2.1B).\n"
                      << "        Solution: Use more MPI ranks to reduce per-rank DOFs.\n"
                      << "        Current global DOFs: " << global_tdofs << "\n\n";
        }
        MPI_Abort(comm, 1);
    }

    // Warning for approaching limit (80%)
    if (local_tdofs > INT_MAX_VAL * 0.8 && rank == 0) {
        std::cerr << "\n[WARNING] Local DOFs (" << local_tdofs
                  << ") approaching 32-bit limit.\n"
                  << "          Consider using more MPI ranks for safety margin.\n\n";
    }
}

}  // anonymous namespace

// =============================================================================
// Configuration Methods
// =============================================================================

template<int Dim>
typename SimulationFacade<Dim>::DerivedType&
SimulationFacade<Dim>::LoadConfig(const std::string& yaml_file) {
    config_ = std::make_unique<YamlConfig>(yaml_file);

    // Check validation result
    if (!config_->IsValid()) {
        MFEM_ABORT("Invalid configuration file: " + config_->GetValidationError());
    }

    // Create output directory (rank 0 only, then barrier)
    std::string output_dir = config_->GetOutputDirectory();
    if (!output_dir.empty()) {
        if (rank_ == 0) {
            CreateDirectory(output_dir);
        }
        MPI_Barrier(comm_);
    }

    // Extract common parameters and set in runner
    real_t dt = config_->GetDt();
    int nt = config_->GetNumSteps();
    runner_.SetDt(dt);
    runner_.SetNumSteps(nt);

    // Set order in components
    components_.SetOrder(config_->GetOrder());

    return *this;
}

template<int Dim>
typename SimulationFacade<Dim>::DerivedType&
SimulationFacade<Dim>::LoadConfig(std::unique_ptr<YamlConfig> config) {
    config_ = std::move(config);

    // Check validation result
    if (!config_->IsValid()) {
        MFEM_ABORT("Invalid configuration file: " + config_->GetValidationError());
    }

    // Create output directory (rank 0 only, then barrier)
    std::string output_dir = config_->GetOutputDirectory();
    if (!output_dir.empty()) {
        if (rank_ == 0) {
            CreateDirectory(output_dir);
        }
        MPI_Barrier(comm_);
    }

    // Extract common parameters and set in runner
    real_t dt = config_->GetDt();
    int nt = config_->GetNumSteps();
    runner_.SetDt(dt);
    runner_.SetNumSteps(nt);

    // Set order in components
    components_.SetOrder(config_->GetOrder());

    return *this;
}

// =============================================================================
// Mesh Setup Methods
// =============================================================================

template<int Dim>
typename SimulationFacade<Dim>::DerivedType&
SimulationFacade<Dim>::SetupMeshFromConfig() {
    MFEM_VERIFY(config_, "Config must be loaded before SetupMeshFromConfig");

    constexpr int dim = Dim;
    int order = components_.Order();

    // Load mesh via config loader
    std::unique_ptr<ParMesh> mesh(LoadParMesh(*config_, comm_));

    // Set high-order mesh curvature for libCEED compatibility
    mesh->SetCurvature(order, true, dim, mfem::Ordering::byNODES);

    // Save mesh if enabled in config (vis.mesh: true)
    if (config_->GetMeshSave()) {
        std::string vis_dir = config_->GetVisDirectory();
        mkdir(vis_dir.c_str(), 0755);
        mesh->SaveAsOne((vis_dir + "/mesh.mesh").c_str());
    }

    // Set mesh in components
    components_.SetMesh(std::move(mesh));

    // Create FE spaces based on material type
    std::string mat_type_str = config_->GetMaterialType();
    SEM::MaterialType mat_type = StringToMaterialType(mat_type_str);
    DomainType domain = GetDomainFromMaterial(mat_type);
    int vdim = (domain == DomainType::Solid) ? dim : 1;

    components_.CreateFESpaces(order, vdim);

    mesh_ready_ = true;

    return *this;
}

// =============================================================================
// Material Setup Methods
// =============================================================================

template<int Dim>
typename SimulationFacade<Dim>::DerivedType&
SimulationFacade<Dim>::SetupMaterialFromConfig() {
    MFEM_VERIFY(config_, "Config must be loaded before SetupMaterialFromConfig");
    MFEM_VERIFY(mesh_ready_, "Mesh must be set up before material");

    constexpr int dim = Dim;
    const int order = components_.Order();
    const int ngll = order + 1;
    IntegrationRules gll_rules(0, Quadrature1D::GaussLobatto);

    // Create tensor product GLL integration rule
    const int exact = 2 * order - 1;
    IntegrationRule ir_1d = gll_rules.Get(Geometry::SEGMENT, exact);

    IntegrationRule ir;
    if constexpr (dim == 2) {
        ir = IntegrationRule(ir_1d, ir_1d);
    } else {
        ir = IntegrationRule(ir_1d, ir_1d, ir_1d);
    }

    // Load material configuration from YAML
    MaterialConfig mat_config = LoadMaterialConfig(*config_);
    // Dispatch (MaterialType → concrete class) is centralized in MaterialFactory.
    // It also applies ApplyAttenuationCorrection() so that unrelaxed moduli are
    // in place before operator setup.
    material_ = SEM::material::CreateMaterial(
        mat_config, components_.FESScalar(), ir, dim);

    material_ready_ = true;

    return *this;
}

// =============================================================================
// Operator Setup Methods
// =============================================================================

template<int Dim>
typename SimulationFacade<Dim>::DerivedType&
SimulationFacade<Dim>::SetupOperatorFromConfig() {
    MFEM_VERIFY(config_, "Config must be loaded before SetupOperatorFromConfig");
    MFEM_VERIFY(mesh_ready_, "Mesh must be set up before operator");
    MFEM_VERIFY(material_ready_, "Material must be set up before operator");

    constexpr int dim = Dim;
    int order = components_.Order();

    // Build operator config
    OperatorConfig op_config;
    op_config.dt = runner_.Dt();

    // Setup damping from config
    ABCConfig abc = config_->GetABCConfig();
    if (!abc.sides.empty()) {
        Array<int> abc_attrs = GetABCBoundaryAttributes(abc, dim);
        if (abc_attrs.Size() > 0) {
            std::vector<real_t> lengths, alphas;
            if constexpr (dim == 2) {
                lengths = {abc.thickness, abc.thickness};
                alphas = {abc.alpha, abc.alpha};
            } else {
                lengths = {abc.thickness, abc.thickness, abc.thickness};
                alphas = {abc.alpha, abc.alpha, abc.alpha};
            }
            std::vector<int> attrs_vec(abc_attrs.GetData(),
                                       abc_attrs.GetData() + abc_attrs.Size());
            op_config.damping = DampingConfig(lengths, alphas, attrs_vec);
        }
    }

    // Setup Dirichlet BC (zero-pressure free surface for fluid/acoustic).
    // Gated on domain rather than specific MaterialType so that future acoustic
    // models pick this up automatically. Elastic operators currently do not
    // implement Dirichlet BC (see WaveOperator::SetupDirichletBC default).
    if (material_->GetDomainType() == DomainType::Fluid) {
        Array<int> dirichlet_attrs = GetDirichletBoundaryAttributes(*config_, dim);
        if (dirichlet_attrs.Size() > 0) {
            // True DOFs (for RHS zeroing in ApplyBoundaryConditions)
            op_config.dirichlet_tdof = GetDirichletTrueDofs(components_.FES(), dirichlet_attrs);

            // Local DOFs (for state vector enforcement in EnforceDirichletBC)
            ParMesh* pmesh = components_.FES().GetParMesh();
            int max_bdr = pmesh->bdr_attributes.Size() > 0
                          ? pmesh->bdr_attributes.Max() : 0;
            if (max_bdr > 0) {
                Array<int> ess_bdr(max_bdr);
                ess_bdr = 0;
                for (int i = 0; i < dirichlet_attrs.Size(); i++) {
                    int attr = dirichlet_attrs[i];
                    if (attr > 0 && attr <= max_bdr) {
                        ess_bdr[attr - 1] = 1;
                    }
                }
                Array<int> ess_vdofs;
                components_.FES().GetEssentialVDofs(ess_bdr, ess_vdofs);
                for (int i = 0; i < ess_vdofs.Size(); i++) {
                    if (ess_vdofs[i] == -1) {
                        op_config.dirichlet_ldof.Append(i);
                    }
                }
            }
        }
    }

    // Create operator using type-safe factory functions.
    // Dispatched by domain (Solid→Elastic, Fluid→Acoustic) so that new solid
    // materials (VTI, TTI, Orthotropic, Poroelastic, ...) route to the elastic
    // operator factory automatically without editing this switch.
    const DomainType domain = material_->GetDomainType();
    if constexpr (Dim == 2) {
        if (domain == DomainType::Solid) {
            op_ = CreateElasticOperator2D(
                components_.FES(), order,
                static_cast<const ElasticMaterialBase2D&>(*material_),
                op_config);
        } else {  // Fluid
            op_ = CreateAcousticOperator2D(
                components_.FES(), order,
                static_cast<const AcousticMaterialBase2D&>(*material_),
                op_config);
        }
    } else {
        if (domain == DomainType::Solid) {
            op_ = CreateElasticOperator3D(
                components_.FES(), order,
                static_cast<const ElasticMaterialBase3D&>(*material_),
                op_config);
        } else {  // Fluid
            op_ = CreateAcousticOperator3D(
                components_.FES(), order,
                static_cast<const AcousticMaterialBase3D&>(*material_),
                op_config);
        }
    }

    // Create time integrator
    integrator_ = std::make_unique<NewmarkCentralDifference>();

    operator_ready_ = true;

    return *this;
}

// =============================================================================
// Source Setup Methods
// =============================================================================

template<int Dim>
typename SimulationFacade<Dim>::DerivedType&
SimulationFacade<Dim>::SetupSourcesFromConfig() {
    MFEM_VERIFY(config_, "Config must be loaded before SetupSourcesFromConfig");
    MFEM_VERIFY(mesh_ready_, "Mesh must be set up before sources");
    MFEM_VERIFY(operator_ready_, "Operator must be set up before sources");

    constexpr int dim = Dim;
    int nt = runner_.NumSteps();
    real_t dt = runner_.Dt();

    // Determine domain from material type
    DomainType domain = GetDomainFromMaterial(material_->GetType());

    if (domain == DomainType::Solid) {
        // Elastic/Viscoelastic sources
        if constexpr (dim == 2) {
            SourceConfig::Config2D src_config = LoadSourceConfig2D(*config_);
            sources_ = PointSourceCollection::FromConfig(
                src_config, &components_.FES(), nt, dt, comm_);
        } else {
            SourceConfig::Config3D src_config = LoadSourceConfig3D(*config_);
            sources_ = PointSourceCollection::FromConfig(
                src_config, &components_.FES(), nt, dt, comm_);
        }
    } else {
        // Acoustic sources
        // Note: For viscoacoustic, material_.Kappa() is already corrected to unrelaxed
        // value in SetupMaterialFromConfig() via ApplyAttenuationCorrection()
        if constexpr (dim == 2) {
            const auto& acoustic_mat = static_cast<const AcousticMaterialBase2D&>(*material_);
            SourceConfig::Config2D src_config = LoadSourceConfig2D(*config_);
            sources_ = PointSourceCollection::FromConfigAcoustic(
                src_config, &components_.FES(), acoustic_mat.Kappa(), nt, dt, comm_);
        } else {
            const auto& acoustic_mat = static_cast<const AcousticMaterialBase3D&>(*material_);
            SourceConfig::Config3D src_config = LoadSourceConfig3D(*config_);
            sources_ = PointSourceCollection::FromConfigAcoustic(
                src_config, &components_.FES(), acoustic_mat.Kappa(), nt, dt, comm_);
        }
    }

    op_->SetupSource(*sources_);
    sources_ready_ = true;

    return *this;
}

// =============================================================================
// Receiver Setup Methods
// =============================================================================

template<int Dim>
typename SimulationFacade<Dim>::DerivedType&
SimulationFacade<Dim>::SetupReceiversFromConfig() {
    MFEM_VERIFY(config_, "Config must be loaded before SetupReceiversFromConfig");
    MFEM_VERIFY(mesh_ready_, "Mesh must be set up before receivers");
    MFEM_VERIFY(material_ready_, "Material must be set up before receivers");

    // Skip receiver setup if no receivers section (inversion mode)
    if (!config_->HasReceivers()) {
        return *this;
    }

    int nt = runner_.NumSteps();
    real_t dt = runner_.Dt();

    // Determine domain from material type
    DomainType domain = GetDomainFromMaterial(material_->GetType());

    receivers_.reset(LoadReceivers(*config_, components_.FES(), &comm_, domain, nt, dt));

    if (receivers_) {
        receivers_->SetFields(&components_.U(), &components_.V(), &components_.A());
        receivers_ready_ = true;

        // Set output configuration for Save()
        std::vector<std::string> recv_formats = config_->GetReceiverOutputFormats();
        std::string recv_filename = "";
        for (const auto& f : recv_formats) {
            if (f == "hdf5" || f == "su") {
                recv_filename = config_->GetReceiverOutputFilename();
                break;
            }
        }
        receivers_->SetOutputConfig(recv_formats, config_->GetOutputDirectory(), recv_filename);

        // Set receivers in IO component
        io_.SetReceivers(receivers_.get());

        // Save filtered receivers log if any were filtered
        if (receivers_->HasFilteredReceivers()) {
            std::string output_dir = config_->GetOutputDirectory();
            receivers_->SaveFilteredLog(output_dir + "/filtered_receivers.txt");
        }
    }

    return *this;
}

// =============================================================================
// Combined Setup
// =============================================================================

template<int Dim>
typename SimulationFacade<Dim>::DerivedType&
SimulationFacade<Dim>::SetupFromConfig() {
    MFEM_VERIFY(config_, "Config must be loaded before SetupFromConfig");

    {
        PROFILE_REGION("Setup:MeshFromConfig");
        SetupMeshFromConfig();
    }
    memory_report_.Record("MeshLoaded");

    {
        PROFILE_REGION("Setup:MaterialFromConfig");
        SetupMaterialFromConfig();
    }
    memory_report_.Record("MaterialSetup");

    {
        PROFILE_REGION("Setup:OperatorFromConfig");
        SetupOperatorFromConfig();
    }
    memory_report_.Record("OperatorBuilt");

    // Validate DOF sizes before kernel allocations (gather_map uses int)
    ValidateLocalDOFSize(components_.FES(), comm_, rank_);

    {
        PROFILE_REGION("Setup:SourcesFromConfig");
        SetupSourcesFromConfig();
    }
    {
        PROFILE_REGION("Setup:ReceiversFromConfig");
        SetupReceiversFromConfig();
    }
    memory_report_.Record("SourceReceiverSetup");

    // Check CFL condition using factor from config (warn but don't abort by default)
    // Users can call CheckCFL(cfl_factor, true) to abort on violation
    CheckCFL(config_->GetCflFactor(), false);

    // Check wavelength sampling using mesh.max_freq and mesh.ppw from config
    real_t f_max = config_->GetMaxFreq();
    real_t ppw_required = config_->GetPPW();
    CheckWavelengthSampling(f_max, ppw_required, true);

    // Free setup-only resources (mass integrator, mass coefficient) to save memory
    if (op_) {
        op_->FreeSetupResources();
    }
    memory_report_.Record("SetupComplete");

    return *this;
}

// =============================================================================
// Device Initialization
// =============================================================================

template<int Dim>
void SimulationFacade<Dim>::DeviceInit() {
    PROFILE_REGION("DeviceInit");

    // 1. Solution vectors
    components_.U().UseDevice(true);
    components_.V().UseDevice(true);
    components_.A().UseDevice(true);

    // 2. Wave operator (includes integrators, geometry, DOFs, mass/damping)
    if (op_) {
        // Call the appropriate DeviceInit based on dimension
        if constexpr (Dim == 3) {
            auto* op3d = dynamic_cast<WaveOperatorBase3D*>(op_.get());
            if (op3d) {
                op3d->DeviceInit();
            }
        } else {
            auto* op2d = dynamic_cast<WaveOperatorBase2D*>(op_.get());
            if (op2d) {
                op2d->DeviceInit();
            }
        }
    }

    // 3. Sources
    if (sources_) {
        sources_->EnableDevice();
        sources_->SyncToDevice();
    }

    // 4. Receivers (GPU recording buffers)
    if (receivers_) {
        int buffer_steps = config_ ? config_->GetReceiverBufferSteps() : 0;
        receivers_->DeviceInit(buffer_steps);
    }

}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

template SimulationFacade<2>::DerivedType& SimulationFacade<2>::LoadConfig(const std::string&);
template SimulationFacade<3>::DerivedType& SimulationFacade<3>::LoadConfig(const std::string&);
template SimulationFacade<2>::DerivedType& SimulationFacade<2>::LoadConfig(std::unique_ptr<YamlConfig>);
template SimulationFacade<3>::DerivedType& SimulationFacade<3>::LoadConfig(std::unique_ptr<YamlConfig>);

template SimulationFacade<2>::DerivedType& SimulationFacade<2>::SetupMeshFromConfig();
template SimulationFacade<3>::DerivedType& SimulationFacade<3>::SetupMeshFromConfig();

template SimulationFacade<2>::DerivedType& SimulationFacade<2>::SetupMaterialFromConfig();
template SimulationFacade<3>::DerivedType& SimulationFacade<3>::SetupMaterialFromConfig();

template SimulationFacade<2>::DerivedType& SimulationFacade<2>::SetupOperatorFromConfig();
template SimulationFacade<3>::DerivedType& SimulationFacade<3>::SetupOperatorFromConfig();

template SimulationFacade<2>::DerivedType& SimulationFacade<2>::SetupSourcesFromConfig();
template SimulationFacade<3>::DerivedType& SimulationFacade<3>::SetupSourcesFromConfig();

template SimulationFacade<2>::DerivedType& SimulationFacade<2>::SetupReceiversFromConfig();
template SimulationFacade<3>::DerivedType& SimulationFacade<3>::SetupReceiversFromConfig();

template SimulationFacade<2>::DerivedType& SimulationFacade<2>::SetupFromConfig();
template SimulationFacade<3>::DerivedType& SimulationFacade<3>::SetupFromConfig();

template void SimulationFacade<2>::DeviceInit();
template void SimulationFacade<3>::DeviceInit();

}  // namespace SEM
