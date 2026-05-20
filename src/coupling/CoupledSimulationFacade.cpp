// Fluid-solid coupled simulation driver. See header for the design overview.

#include "coupling/CoupledSimulationFacade.hpp"

#include "config/ConfigLoaders.hpp"
#include "material/MaterialFactory.hpp"
#include "material/AcousticMaterialBase.hpp"
#include "material/ElasticMaterialBase.hpp"
#include "material/MaterialField.hpp"
#include "operator/OperatorFactory.hpp"
#include "operator/WaveOperatorBase2D.hpp"
#include "operator/WaveOperatorBase3D.hpp"
#include "srcrecv/SrcRecvLoaders.hpp"
#include "util/CFLReport.hpp"
#include "util/ElementMetrics.hpp"
#include "util/Profiler.hpp"
#include "util/WavelengthReport.hpp"

#include <mfem.hpp>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>

#ifdef __linux__
#include <sys/stat.h>
#else
#include <direct.h>
#endif

namespace SEM {

using namespace mfem;

template<int Dim>
CoupledSimulationFacade<Dim>::CoupledSimulationFacade(MPI_Comm comm)
    : comm_(comm),
      fluid_comp_(comm), solid_comp_(comm),
      fluid_runner_(comm), solid_runner_(comm)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &num_procs_);
}

template<int Dim>
CoupledSimulationFacade<Dim>::~CoupledSimulationFacade() = default;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

template<int Dim>
CoupledSimulationFacade<Dim>&
CoupledSimulationFacade<Dim>::LoadConfig(const std::string& yaml_file)
{
    auto cfg = std::make_unique<YamlConfig>(yaml_file);
    MFEM_VERIFY(cfg->IsValid(),
                "Failed to load coupled simulation config '" << yaml_file
                << "': " << cfg->GetValidationError());
    return LoadConfig(std::move(cfg));
}

template<int Dim>
CoupledSimulationFacade<Dim>&
CoupledSimulationFacade<Dim>::LoadConfig(std::unique_ptr<YamlConfig> config)
{
    MFEM_VERIFY(config, "null YamlConfig passed to LoadConfig");
    MFEM_VERIFY(config->IsCoupledMaterial(),
                "CoupledSimulationFacade requires material.type: coupled");
    config_      = std::move(config);
    coupled_cfg_ = LoadCoupledMaterialConfig(*config_);

    // Create the top-level output directory (rank 0, everyone waits).
    // Mirrors SimulationFacade<Dim>::LoadConfig so that log.txt / summary.txt
    // / wavefield subdirs can be opened without the user pre-creating the
    // directory. The GLVis writer still creates its own nested subdirs.
    const std::string outdir = config_->GetOutputDirectory();
    if (!outdir.empty()) {
        if (IsRoot()) {
#ifdef __linux__
            (void)::mkdir(outdir.c_str(), 0755);   // ignore EEXIST
#else
            (void)::_mkdir(outdir.c_str());
#endif
        }
        MPI_Barrier(comm_);
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

namespace {

/// Build the tensor-product GLL rule used by the material factory. Mirrors
/// the recipe in SimulationFacade<Dim>::SetupMaterialFromConfig so both
/// single-physics and coupled simulations use the same quadrature.
IntegrationRule MakeGLLIntegrationRule(int order, int dim)
{
    IntegrationRules gll_rules(0, Quadrature1D::GaussLobatto);
    const int exact = 2 * order - 1;
    IntegrationRule ir_1d = gll_rules.Get(Geometry::SEGMENT, exact);
    if (dim == 2) return IntegrationRule(ir_1d, ir_1d);
    return IntegrationRule(ir_1d, ir_1d, ir_1d);
}

template<int Dim>
void SetupSubmesh(SimulationComponents<Dim>& comp,
                  ParMesh& parent,
                  int domain_attr,
                  int order,
                  int vdim)
{
    Array<int> attrs(1);
    attrs[0] = domain_attr;
    // Heap-allocate so SimulationComponents can take std::unique_ptr<ParMesh>.
    auto* sub = new ParSubMesh(ParSubMesh::CreateFromDomain(parent, attrs));
    sub->SetCurvature(order, true, Dim, mfem::Ordering::byNODES);
    comp.SetOrder(order);
    comp.SetMesh(std::unique_ptr<ParMesh>(sub));
    comp.CreateFESpaces(order, vdim);
}

}  // namespace

template<int Dim>
CoupledSimulationFacade<Dim>&
CoupledSimulationFacade<Dim>::SetupFromConfig()
{
    MFEM_VERIFY(config_,
                "LoadConfig must be called before SetupFromConfig");
    MFEM_VERIFY(!setup_done_,
                "CoupledSimulationFacade::SetupFromConfig already called");

    // 1. Parent mesh (same path as single-physics simulations).
    parent_mesh_.reset(LoadParMesh(*config_, comm_));
    MFEM_VERIFY(parent_mesh_, "LoadParMesh returned null for coupled config");

    // Sanity: parent must have BOTH element attributes that the coupled
    // config references. Misconfiguration here produces a deeply confusing
    // error later (empty submesh on all ranks), so catch it up front.
    {
        bool has_fluid = false, has_solid = false;
        for (int a = 0; a < parent_mesh_->attributes.Size(); ++a) {
            const int attr = parent_mesh_->attributes[a];
            if (attr == coupled_cfg_.fluid_attribute) has_fluid = true;
            if (attr == coupled_cfg_.solid_attribute) has_solid = true;
        }
        // attributes is the local set of attributes observed on this rank;
        // aggregate via MPI so ranks with no fluid/solid elements don't abort.
        int f = has_fluid ? 1 : 0;
        int s = has_solid ? 1 : 0;
        int f_any = 0, s_any = 0;
        MPI_Allreduce(&f, &f_any, 1, MPI_INT, MPI_LOR, comm_);
        MPI_Allreduce(&s, &s_any, 1, MPI_INT, MPI_LOR, comm_);
        MFEM_VERIFY(f_any,
                    "parent mesh has no elements with fluid attribute "
                    << coupled_cfg_.fluid_attribute);
        MFEM_VERIFY(s_any,
                    "parent mesh has no elements with solid attribute "
                    << coupled_cfg_.solid_attribute);
    }

    const int order = config_->GetOrder();
    MFEM_VERIFY(order > 0, "simulation.order must be set (got " << order << ")");

    // 2. Submesh + FE spaces per domain.
    //    Fluid: scalar field (displacement potential φ), vdim = 1.
    //    Solid: vector displacement u_s, vdim = Dim.
    SetupSubmesh<Dim>(fluid_comp_, *parent_mesh_,
                      coupled_cfg_.fluid_attribute, order, /*vdim=*/1);
    SetupSubmesh<Dim>(solid_comp_, *parent_mesh_,
                      coupled_cfg_.solid_attribute, order, /*vdim=*/Dim);

    // 3. Materials — go through the same factory as single-physics paths
    //    so attenuation correction and future MaterialType additions apply
    //    automatically.
    const IntegrationRule ir = MakeGLLIntegrationRule(order, Dim);
    fluid_material_ = material::CreateMaterial(
        coupled_cfg_.fluid, fluid_comp_.FESScalar(), ir, Dim);
    solid_material_ = material::CreateMaterial(
        coupled_cfg_.solid, solid_comp_.FESScalar(), ir, Dim);

    // Physical sanity: the MaterialFactory returns concrete types matching
    // the YAML. Re-check domain assignment here so typos in the YAML fail
    // loudly rather than producing a silently inverted coupling later.
    MFEM_VERIFY(fluid_material_->GetDomainType() == DomainType::Fluid,
                "fluid sub-material resolves to domain "
                << (int)fluid_material_->GetDomainType()
                << " — material.fluid.type must be acoustic");
    MFEM_VERIFY(solid_material_->GetDomainType() == DomainType::Solid,
                "solid sub-material resolves to domain "
                << (int)solid_material_->GetDomainType()
                << " — material.solid.type must be elastic");

    // 4. Operators. Both sides share a single dt read from simulation
    //    so the time loop stays aligned; the effective CFL is the min of
    //    the two sides.
    dt_ = config_->GetDt();
    nt_ = config_->GetNumSteps();
    MFEM_VERIFY(dt_ > 0.0, "simulation.dt must be positive");
    MFEM_VERIFY(nt_ > 0,   "simulation.steps must be positive");

    // Dirichlet handling for the coupled case is non-trivial (Doc §5.2.2
    // reminder). Rules:
    //   * Only attributes actually present on the submesh qualify; parent
    //     bdrs on the "other" half (e.g. a solid-side top face) would silently
    //     match nothing, but we strip them up front for diagnostic clarity.
    //   * The **auto-generated interface attribute** (set-diff against parent)
    //     MUST NEVER be Dirichlet; it IS the coupling surface. Silently
    //     filtering it out here stops a misconfigured YAML from turning the
    //     interface into a rigid/zero-pressure wall and masking the coupling.
    //   * Only the acoustic operator currently honors dirichlet_tdof/ldof;
    //     for the elastic side we leave it empty (matches SimulationFacade).
    auto autogen_bdrs = [&](const ParMesh& p, const ParMesh& s) {
        std::set<int> ps, ret;
        for (int i = 0; i < p.bdr_attributes.Size(); ++i) ps.insert(p.bdr_attributes[i]);
        for (int i = 0; i < s.bdr_attributes.Size(); ++i) {
            int a = s.bdr_attributes[i];
            if (!ps.count(a)) ret.insert(a);
        }
        return ret;
    };
    auto as_set = [](const mfem::Array<int>& arr) {
        std::set<int> s;
        for (int i = 0; i < arr.Size(); ++i) s.insert(arr[i]);
        return s;
    };

    OperatorConfig fluid_op_cfg, solid_op_cfg;
    fluid_op_cfg.dt = dt_;
    solid_op_cfg.dt = dt_;

    // Cerjan absorbing boundary.
    //
    // Cerjan is a **volumetric multiplicative taper**: every state DOF
    // whose physical position lies within `thickness` of one of the
    // boundary attributes listed in `abc_attrs` gets scaled down each
    // timestep. Note: this includes interface DOFs that happen to sit
    // inside a side-wall sponge layer; that is the physically correct
    // behavior (the sponge eats waves regardless of which material
    // they're propagating in), and it applies symmetrically on both
    // fluid and solid sides so the interface exchange stays balanced.
    //
    // The auto-generated interface attribute is never in `abc.sides` —
    // users specify outer faces (top/bottom/left/right/front/back) in
    // YAML and the interface is auto-detected — so no explicit
    // interface filter is needed.
    //
    // The only filter we do apply is "attribute actually exists on THIS
    // submesh": e.g., a bottom-face attr only shows up on the fluid
    // submesh in a water-over-rock setup, so passing it to the solid
    // side's DampingConfig would be a no-op. We strip it for clarity
    // and to avoid spurious diagnostic output, not for correctness.
    //
    // Dirichlet × Cerjan overlap: when a DOF is BOTH on a Dirichlet
    // boundary AND inside a Cerjan sponge layer, the Step() order
    //   predictor → EnforceDirichletBC → ExplicitSolve → corrector
    //     → ApplyCerjanTaper → EnforceDirichletBC
    // guarantees Dirichlet wins: the taper multiplies whatever state
    // Cerjan touches, then the final EnforceDirichletBC re-zeros the
    // Dirichlet DOFs. Damped * 0 = 0, so the physics is consistent;
    // the taper operation on Dirichlet DOFs is wasted work but
    // harmless. We don't filter Dirichlet out of the damping list for
    // this reason; the ordering alone is sufficient.
    {
        ABCConfig abc = config_->GetABCConfig();
        if (!abc.sides.empty() && abc.thickness > 0.0) {
            const mfem::Array<int> abc_attrs =
                GetABCBoundaryAttributes(abc, Dim);
            if (abc_attrs.Size() > 0) {
                std::vector<real_t> lengths, alphas;
                if constexpr (Dim == 2) {
                    lengths = {abc.thickness, abc.thickness};
                    alphas  = {abc.alpha, abc.alpha};
                } else {
                    lengths = {abc.thickness, abc.thickness, abc.thickness};
                    alphas  = {abc.alpha, abc.alpha, abc.alpha};
                }
                auto filter_to_submesh = [&](const ParMesh& sub_mesh) {
                    const auto sub_bdrs = as_set(sub_mesh.bdr_attributes);
                    std::vector<int> out;
                    for (int i = 0; i < abc_attrs.Size(); ++i) {
                        const int a = abc_attrs[i];
                        if (sub_bdrs.count(a)) out.push_back(a);
                    }
                    return out;
                };
                const auto fluid_attrs = filter_to_submesh(fluid_comp_.Mesh());
                const auto solid_attrs = filter_to_submesh(solid_comp_.Mesh());
                if (!fluid_attrs.empty()) {
                    fluid_op_cfg.damping =
                        DampingConfig(lengths, alphas, fluid_attrs);
                }
                if (!solid_attrs.empty()) {
                    solid_op_cfg.damping =
                        DampingConfig(lengths, alphas, solid_attrs);
                }
            }
        }
    }
    {
        auto dirichlet_full = GetDirichletBoundaryAttributes(*config_, Dim);
        const auto fluid_auto   = autogen_bdrs(*parent_mesh_, fluid_comp_.Mesh());
        const auto fluid_bdrs   = as_set(fluid_comp_.Mesh().bdr_attributes);
        mfem::Array<int> fluid_dirichlet;
        for (int i = 0; i < dirichlet_full.Size(); ++i) {
            int a = dirichlet_full[i];
            if (fluid_bdrs.count(a) && !fluid_auto.count(a)) {
                fluid_dirichlet.Append(a);
            }
        }
        if (fluid_dirichlet.Size() > 0) {
            fluid_op_cfg.dirichlet_tdof =
                GetDirichletTrueDofs(fluid_comp_.FES(), fluid_dirichlet);
            ParMesh& fmesh = fluid_comp_.Mesh();
            const int max_bdr = fmesh.bdr_attributes.Size() > 0
                              ? fmesh.bdr_attributes.Max() : 0;
            if (max_bdr > 0) {
                mfem::Array<int> ess_bdr(max_bdr);
                ess_bdr = 0;
                for (int i = 0; i < fluid_dirichlet.Size(); ++i) {
                    int a = fluid_dirichlet[i];
                    if (a > 0 && a <= max_bdr) ess_bdr[a - 1] = 1;
                }
                mfem::Array<int> ess_vdofs;
                fluid_comp_.FES().GetEssentialVDofs(ess_bdr, ess_vdofs);
                for (int i = 0; i < ess_vdofs.Size(); ++i) {
                    if (ess_vdofs[i] == -1) fluid_op_cfg.dirichlet_ldof.Append(i);
                }
            }
        }
        // Solid side: elastic operator currently ignores dirichlet_*; leave
        // empty. Future: filter analogously if elastic starts honoring it.
    }

    if constexpr (Dim == 2) {
        fluid_op_ = CreateAcousticOperator2D(
            fluid_comp_.FES(), order,
            static_cast<const AcousticMaterialBase2D&>(*fluid_material_),
            fluid_op_cfg);
        solid_op_ = CreateElasticOperator2D(
            solid_comp_.FES(), order,
            static_cast<const ElasticMaterialBase2D&>(*solid_material_),
            solid_op_cfg);
    } else {
        fluid_op_ = CreateAcousticOperator3D(
            fluid_comp_.FES(), order,
            static_cast<const AcousticMaterialBase3D&>(*fluid_material_),
            fluid_op_cfg);
        solid_op_ = CreateElasticOperator3D(
            solid_comp_.FES(), order,
            static_cast<const ElasticMaterialBase3D&>(*solid_material_),
            solid_op_cfg);
    }

    // 5. Integrators + runners. Both sides use the same integrator type so
    //    a per-step comparison against a single-physics reference stays
    //    bit-reproducible when the interface term is later zeroed out.
    fluid_integrator_ = std::make_unique<NewmarkCentralDifference>();
    solid_integrator_ = std::make_unique<NewmarkCentralDifference>();

    fluid_runner_.SetDt(dt_);
    fluid_runner_.SetNumSteps(nt_);
    fluid_runner_.SetT0(config_->GetT0());
    fluid_runner_.SetOperator(fluid_op_.get());
    fluid_runner_.SetIntegrator(fluid_integrator_.get());

    solid_runner_.SetDt(dt_);
    solid_runner_.SetNumSteps(nt_);
    solid_runner_.SetT0(config_->GetT0());
    solid_runner_.SetOperator(solid_op_.get());
    solid_runner_.SetIntegrator(solid_integrator_.get());

    // 6. Sources & receivers (if present in YAML). Routing is type-based:
    //   pressure      -> fluid (acoustic)        — error if put in solid
    //   force / MT    -> solid (elastic)         — error if put in fluid
    //   receiver PS   -> fluid
    //   receiver DISP/VEL/ACC/GRAD -> solid
    // Geometric validation (position actually inside the expected submesh)
    // is delegated to PointFinder inside PointSourceCollection / ReceiverArray;
    // a type-vs-position mismatch surfaces as a "point not found" error
    // from the per-submesh search.
    //
    // TODO (future): dipole / volumetric force sources in acoustic media
    // (e.g. vertical force in water from a submerged surface vibration).
    // When implemented, forces with an "acoustic" flag / location inside
    // the fluid submesh should route to fluid_sources_ rather than
    // solid_sources_ here. The domain routing rules above assume the
    // current source-type space (pressure, force, moment_tensor) only.
    SetupSourcesFromConfig();
    SetupReceiversFromConfig();

    // 7. Interface cache — auto-detected attribute + per-quad normals /
    //    jacobianw / DOF lists ready for Step() to consume via the
    //    Apply*/Extract* API. Safe to set up here because both submeshes,
    //    FE spaces and materials are in place.
    iface_.Setup(*parent_mesh_,
                 static_cast<ParSubMesh&>(fluid_comp_.Mesh()),
                 static_cast<ParSubMesh&>(solid_comp_.Mesh()),
                 coupled_cfg_.fluid_attribute,
                 coupled_cfg_.solid_attribute,
                 fluid_comp_.FES(), solid_comp_.FES());

    setup_done_ = true;
    return *this;
}

// ---------------------------------------------------------------------------
// Sources & Receivers
// ---------------------------------------------------------------------------

namespace {

/// Split a flat SourceConfig::Config3D into (fluid, solid) sub-configs by
/// source type. Pressures → fluid; forces / moment tensors → solid.
std::pair<SourceConfig::Config3D, SourceConfig::Config3D>
SplitSources3D(const SourceConfig::Config3D& src)
{
    SourceConfig::Config3D fluid_cfg, solid_cfg;
    fluid_cfg.pressures      = src.pressures;
    solid_cfg.forces         = src.forces;
    solid_cfg.moment_tensors = src.moment_tensors;
    return {fluid_cfg, solid_cfg};
}

std::pair<SourceConfig::Config2D, SourceConfig::Config2D>
SplitSources2D(const SourceConfig::Config2D& src)
{
    SourceConfig::Config2D fluid_cfg, solid_cfg;
    fluid_cfg.pressures      = src.pressures;
    solid_cfg.forces         = src.forces;
    solid_cfg.moment_tensors = src.moment_tensors;
    return {fluid_cfg, solid_cfg};
}

}  // namespace

template<int Dim>
void CoupledSimulationFacade<Dim>::SetupSourcesFromConfig()
{
    if constexpr (Dim == 2) {
        auto src_cfg = LoadSourceConfig2D(*config_);
        auto [fluid_cfg, solid_cfg] = SplitSources2D(src_cfg);
        const auto& acoustic_mat =
            static_cast<const AcousticMaterialBase2D&>(*fluid_material_);
        fluid_sources_ = PointSourceCollection::FromConfigAcoustic(
            fluid_cfg, &fluid_comp_.FES(), acoustic_mat.Kappa(),
            nt_, dt_, comm_);
        solid_sources_ = PointSourceCollection::FromConfig(
            solid_cfg, &solid_comp_.FES(), nt_, dt_, comm_);
    } else {
        auto src_cfg = LoadSourceConfig3D(*config_);
        auto [fluid_cfg, solid_cfg] = SplitSources3D(src_cfg);
        const auto& acoustic_mat =
            static_cast<const AcousticMaterialBase3D&>(*fluid_material_);
        fluid_sources_ = PointSourceCollection::FromConfigAcoustic(
            fluid_cfg, &fluid_comp_.FES(), acoustic_mat.Kappa(),
            nt_, dt_, comm_);
        solid_sources_ = PointSourceCollection::FromConfig(
            solid_cfg, &solid_comp_.FES(), nt_, dt_, comm_);
    }

    if (fluid_sources_ && fluid_sources_->NumSources() > 0) {
        fluid_op_->SetupSource(*fluid_sources_);
    }
    if (solid_sources_ && solid_sources_->NumSources() > 0) {
        solid_op_->SetupSource(*solid_sources_);
    }
}

template<int Dim>
void CoupledSimulationFacade<Dim>::SetupReceiversFromConfig()
{
    if (!config_->HasReceivers()) return;

    // LoadReceivers filters receiver.types against domain, so we call it
    // twice to pick up {PS} on the fluid side and {DISP/VEL/ACC/GRAD} on
    // the solid side. Receivers whose types are all filtered out simply
    // end up in a zero-length array (not an error).
    fluid_receivers_.reset(LoadReceivers(
        *config_, fluid_comp_.FES(), &comm_, DomainType::Fluid, nt_, dt_));
    solid_receivers_.reset(LoadReceivers(
        *config_, solid_comp_.FES(), &comm_, DomainType::Solid, nt_, dt_));

    if (fluid_receivers_) {
        fluid_receivers_->SetFields(
            &fluid_comp_.U(), &fluid_comp_.V(), &fluid_comp_.A());
        const std::vector<std::string> fmt = config_->GetReceiverOutputFormats();
        fluid_receivers_->SetOutputConfig(
            fmt, config_->GetOutputDirectory(),
            config_->GetReceiverOutputFilename() + "_fluid");
    }
    if (solid_receivers_) {
        solid_receivers_->SetFields(
            &solid_comp_.U(), &solid_comp_.V(), &solid_comp_.A());
        const std::vector<std::string> fmt = config_->GetReceiverOutputFormats();
        solid_receivers_->SetOutputConfig(
            fmt, config_->GetOutputDirectory(),
            config_->GetReceiverOutputFilename() + "_solid");
    }

    // Dump per-side filtered receiver logs so users see what got dropped
    // (e.g. fluid stations whose position lies in the solid submesh).
    const std::string out_dir = config_->GetOutputDirectory();
    if (fluid_receivers_ && fluid_receivers_->HasFilteredReceivers()) {
        fluid_receivers_->SaveFilteredLog(out_dir + "/filtered_receivers_fluid.txt");
    }
    if (solid_receivers_ && solid_receivers_->HasFilteredReceivers()) {
        solid_receivers_->SaveFilteredLog(out_dir + "/filtered_receivers_solid.txt");
    }
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

template<int Dim>
void CoupledSimulationFacade<Dim>::Initialize()
{
    MFEM_VERIFY(setup_done_, "Initialize called before SetupFromConfig");
    fluid_runner_.Initialize(fluid_comp_.U(), fluid_comp_.V(), fluid_comp_.A());
    solid_runner_.Initialize(solid_comp_.U(), solid_comp_.V(), solid_comp_.A());
    initialized_once_ = true;
}

template<int Dim>
void CoupledSimulationFacade<Dim>::Reset()
{
    // Both runners own their own t0/step state, so each side must be reset
    // through its own runner. `ResetData()` on receivers zeroes the record
    // buffer so the next run starts with a clean recording.
    fluid_runner_.Reset(fluid_comp_.U(), fluid_comp_.V(), fluid_comp_.A(),
                        fluid_op_.get());
    solid_runner_.Reset(solid_comp_.U(), solid_comp_.V(), solid_comp_.A(),
                        solid_op_.get());

    if (fluid_receivers_) fluid_receivers_->ResetData();
    if (solid_receivers_) solid_receivers_->ResetData();

    // Arm the initialize flag so a subsequent Step() treats the new state
    // as t=T0 rather than racing into the loop body uninitialised.
    initialized_once_ = false;
}

template<int Dim>
void CoupledSimulationFacade<Dim>::DeviceInit()
{
    PROFILE_REGION("CoupledDeviceInit");
    MFEM_VERIFY(setup_done_, "DeviceInit called before SetupFromConfig");

    auto init_side = [&](SimulationComponents<Dim>&         comp,
                         WaveOperator*                      op,
                         PointSourceCollection*             sources,
                         ReceiverArray*                     receivers)
    {
        // 1. State vectors
        comp.U().UseDevice(true);
        comp.V().UseDevice(true);
        comp.A().UseDevice(true);

        // 2. Operator (integrators, geometry, DOFs, mass/damping, rhs)
        if (op) {
            if constexpr (Dim == 3) {
                if (auto* op3d = dynamic_cast<WaveOperatorBase3D*>(op)) {
                    op3d->DeviceInit();
                }
            } else {
                if (auto* op2d = dynamic_cast<WaveOperatorBase2D*>(op)) {
                    op2d->DeviceInit();
                }
            }
        }

        // 3. Sources
        if (sources) {
            sources->EnableDevice();
            sources->SyncToDevice();
        }

        // 4. Receiver GPU recording buffers
        if (receivers) {
            const int buffer_steps = config_ ? config_->GetReceiverBufferSteps() : 0;
            receivers->DeviceInit(buffer_steps);
        }
    };

    init_side(fluid_comp_, fluid_op_.get(),
              fluid_sources_.get(), fluid_receivers_.get());
    init_side(solid_comp_, solid_op_.get(),
              solid_sources_.get(), solid_receivers_.get());

    // NOTE: FluidSolidInterface::Setup has already flipped its cached
    // normals / jacobian_w / DOF arrays and remote-exchange buffers to
    // UseDevice(true), so nothing extra is needed here for the coupling
    // path. See FluidSolidInterface.cpp for the UseDevice(true) calls.
}

template<int Dim>
bool CoupledSimulationFacade<Dim>::Step()
{
    MFEM_VERIFY(initialized_once_, "Step called before Initialize");

    // Record both sides at the current step BEFORE integration (matches
    // SimulationFacade<Dim>::Step), so the first sample is the zero
    // initial state and recorded index == runner step index.
    const int buffer_steps = config_ ? config_->GetReceiverBufferSteps() : 0;
    if (fluid_receivers_) {
        fluid_receivers_->Record(fluid_runner_.CurrentStep(), buffer_steps);
    }
    if (solid_receivers_) {
        solid_receivers_->Record(solid_runner_.CurrentStep(), buffer_steps);
    }

    // Coupling-disabled fallback: no interface found or explicitly turned off → each
    // side steps independently. This is still the code path that backs the
    // no-coupling independence regression.
    if (!coupling_enabled_ || iface_.InterfaceAttribute() < 0) {
        const bool f_more = fluid_runner_.Step(
            fluid_comp_.U(), fluid_comp_.V(), fluid_comp_.A());
        const bool s_more = solid_runner_.Step(
            solid_comp_.U(), solid_comp_.V(), solid_comp_.A());
        return f_more || s_more;
    }

    // -----------------------------------------------------------------
    // Coupled step. Staggered order (Doc §2.5): predictor → fluid
    // solve (+ interface from predictor u_s) → solid solve
    // (+ interface from UPDATED φ_tt) → corrector.
    //
    // We bypass SimulationRunner::Step here so we can inject the
    // interface contributions between the two ExplicitSolves without
    // reassembling a BoundaryLFIntegrator every timestep. The runners
    // still own time/step bookkeeping via SetState.
    // -----------------------------------------------------------------

    const real_t dt      = dt_;
    const real_t half_dt = 0.5 * dt;
    const real_t half_dt2 = 0.5 * dt * dt;
    const real_t t_next  = (fluid_runner_.CurrentStep() + 1) * dt;

    // Operator time (source evaluation) at t+dt, matches NewmarkCD.
    fluid_op_->SetTime(t_next);
    solid_op_->SetTime(t_next);

    // 1. Predictor both sides: u += dt·v + ½dt²·a ; v += ½dt·a.
    auto predict = [&](ParGridFunction& U, ParGridFunction& V,
                       const ParGridFunction& A) {
        const int N = U.Size();
        real_t*       u = U.ReadWrite();
        real_t*       v = V.ReadWrite();
        const real_t* a = A.Read();
        const real_t cdt = dt, ch = half_dt, ch2 = half_dt2;
        mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) {
            u[i] += cdt * v[i] + ch2 * a[i];
            v[i] += ch * a[i];
        });
    };
    predict(fluid_comp_.U(), fluid_comp_.V(), fluid_comp_.A());
    predict(solid_comp_.U(), solid_comp_.V(), solid_comp_.A());

    // Dirichlet on predicted state (matches NewmarkCD pre-solve step).
    fluid_op_->EnforceDirichletBC(fluid_comp_.U(), fluid_comp_.V(),
                                  fluid_comp_.A());
    solid_op_->EnforceDirichletBC(solid_comp_.U(), solid_comp_.V(),
                                  solid_comp_.A());

    // 2. Fluid: pre-coupling RHS → scatter solid→fluid coupling into
    //    the raw rhs_ → finalize (ParallelAssemble + BC + M⁻¹). The
    //    coupling surface integral is added to the unassembled RHS
    //    before the final mass inversion.
    fluid_op_->AssemblePreCouplingRHS(fluid_comp_.U(), dt);
    ParGridFunction* fluid_rhs_pgf = fluid_op_->RhsPtr();
    MFEM_VERIFY(fluid_rhs_pgf,
                "fluid operator must expose RhsPtr() for coupling");
    iface_.ApplySolidToFluidRHS(solid_comp_.U(), *fluid_rhs_pgf);
    fluid_op_->FinalizeAndApplyMass(fluid_comp_.V(), fluid_comp_.A());

    // 2b. Re-enforce the fluid Dirichlet (free-surface p=0) on the
    //     just-computed fluid state BEFORE the solid reads φ̈, i.e.
    //     right after the mass inversion and right before the elastic
    //     solve that consumes φ̈ via coupling. Without this step, at
    //     corners where Γ_fs meets a fluid free surface (e.g. the
    //     vertical sidewalls in the 2D fluid-solid benchmark) the
    //     fluid acceleration can carry a spurious non-zero value
    //     injected by the coupling scatter itself. The leak is
    //     non-physical and accumulates through multiple round-trip
    //     reflections, producing a small per-step bias that grows
    //     into an O(20 %) L2 drift over ~2 s of simulation.
    fluid_op_->EnforceDirichletBC(fluid_comp_.U(), fluid_comp_.V(),
                                  fluid_comp_.A());

    // 3. Solid: same pattern. The fluid→solid coupling uses the UPDATED
    //    fluid acceleration φ̈ = fluid_comp_.A() from step 2 (Doc §2.5 —
    //    "requires this coupling term uses the updated pressure at
    //    time step [t+delta_t]").
    solid_op_->AssemblePreCouplingRHS(solid_comp_.U(), dt);
    ParGridFunction* solid_rhs_pgf = solid_op_->RhsPtr();
    MFEM_VERIFY(solid_rhs_pgf,
                "solid operator must expose RhsPtr() for coupling");
    iface_.ApplyFluidToSolidRHS(fluid_comp_.A(), *solid_rhs_pgf);
    solid_op_->FinalizeAndApplyMass(solid_comp_.V(), solid_comp_.A());

    // 4. Corrector both sides: v += ½dt·a.
    auto correct = [&](ParGridFunction& V, const ParGridFunction& A) {
        const int N = V.Size();
        real_t*       v = V.ReadWrite();
        const real_t* a = A.Read();
        const real_t ch = half_dt;
        mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) {
            v[i] += ch * a[i];
        });
    };
    correct(fluid_comp_.V(), fluid_comp_.A());
    correct(solid_comp_.V(), solid_comp_.A());

    // 5. Post-step housekeeping (matches NewmarkCD tail).
    fluid_op_->ApplyCerjanTaper(fluid_comp_.U(), fluid_comp_.V(),
                                fluid_comp_.A());
    solid_op_->ApplyCerjanTaper(solid_comp_.U(), solid_comp_.V(),
                                solid_comp_.A());
    fluid_op_->EnforceDirichletBC(fluid_comp_.U(), fluid_comp_.V(),
                                  fluid_comp_.A());
    solid_op_->EnforceDirichletBC(solid_comp_.U(), solid_comp_.V(),
                                  solid_comp_.A());

    // Update runner state counters so CurrentStep / Dt / IsFinished
    // queries (e.g. from the receiver record loop) stay consistent.
    const int next_step = fluid_runner_.CurrentStep() + 1;
    fluid_runner_.SetState(next_step, t_next);
    solid_runner_.SetState(next_step, t_next);
    return next_step < nt_;
}

template<int Dim>
void CoupledSimulationFacade<Dim>::Run()
{
    if (!initialized_once_) Initialize();

    // Initialize wavefield writers (one set per submesh). Each writer
    // takes its own mesh + total step count so GLVis / HDF5 output stays
    // separate for fluid and solid sides.
    //
    // Before Init, hand each writer the *parent* mesh bbox so any
    // grid-based writer (GMT) lays its interpolation grid over the
    // full domain, not just the side's submesh. Without this the
    // fluid side's GMT grid covers only the disjoint inclusion
    // bounding-box (e.g. just the "icm" letter rectangle) while the
    // solid side covers the full host, so overlay visualisers see
    // two incompatible grids. Writers that aren't grid-based (GLVis /
    // ParaView) ignore this call via the base-class no-op.
    Vector parent_bb_min, parent_bb_max;
    parent_mesh_->GetBoundingBox(parent_bb_min, parent_bb_max);
    for (auto& w : fluid_wavefield_writers_) {
        w->SetBoundingBoxOverride(parent_bb_min, parent_bb_max);
        w->Init(fluid_comp_.Mesh(), nt_, comm_);
    }
    for (auto& w : solid_wavefield_writers_) {
        w->SetBoundingBoxOverride(parent_bb_min, parent_bb_max);
        w->Init(solid_comp_.Mesh(), nt_, comm_);
    }

    // Time-step loop. Progress + wavefield output happen WITHIN this loop
    // so the same cadence is observed on both domains.
    while (Step()) {
        const int step = fluid_runner_.CurrentStep();
        const real_t t = fluid_runner_.CurrentTime();

        // Per-step progress: per-domain u_min/u_max (min/max over this
        // rank, MPI-reduced inside ProgressLogger::Tick).
        if (fluid_progress_.Enabled()) {
            real_t lo = fluid_comp_.U().Min();
            real_t hi = fluid_comp_.U().Max();
            fluid_progress_.Tick(step, nt_, t, lo, hi);
        }
        if (solid_progress_.Enabled()) {
            real_t lo = solid_comp_.U().Min();
            real_t hi = solid_comp_.U().Max();
            solid_progress_.Tick(step, nt_, t, lo, hi);
        }

        // Wavefield snapshots. Source id is taken from the first source on
        // whichever domain has one (for filename suffixing, same as
        // ForwardSimulation's convention).
        int src_id = -1;
        if (solid_sources_ && solid_sources_->NumSources() > 0) {
            src_id = solid_sources_->GetSource(0)->GetId();
        } else if (fluid_sources_ && fluid_sources_->NumSources() > 0) {
            src_id = fluid_sources_->GetSource(0)->GetId();
        }
        for (auto& w : fluid_wavefield_writers_) {
            if (w->ShouldWrite(step)) {
                w->Write(step, t,
                         &fluid_comp_.U(), &fluid_comp_.V(), &fluid_comp_.A(),
                         src_id);
            }
        }
        for (auto& w : solid_wavefield_writers_) {
            if (w->ShouldWrite(step)) {
                w->Write(step, t,
                         &solid_comp_.U(), &solid_comp_.V(), &solid_comp_.A(),
                         src_id);
            }
        }
    }

    // Final snapshot at the last recorded step.
    {
        const int step = fluid_runner_.CurrentStep();
        const real_t t = fluid_runner_.CurrentTime();
        int src_id = -1;
        if (solid_sources_ && solid_sources_->NumSources() > 0) {
            src_id = solid_sources_->GetSource(0)->GetId();
        } else if (fluid_sources_ && fluid_sources_->NumSources() > 0) {
            src_id = fluid_sources_->GetSource(0)->GetId();
        }
        for (auto& w : fluid_wavefield_writers_) {
            if (w->ShouldWrite(step)) {
                w->Write(step, t,
                         &fluid_comp_.U(), &fluid_comp_.V(), &fluid_comp_.A(),
                         src_id);
            }
        }
        for (auto& w : solid_wavefield_writers_) {
            if (w->ShouldWrite(step)) {
                w->Write(step, t,
                         &solid_comp_.U(), &solid_comp_.V(), &solid_comp_.A(),
                         src_id);
            }
        }
    }

    // Close writers. Same pattern as ForwardSimulation::Run.
    for (auto& w : fluid_wavefield_writers_) w->Finalize();
    for (auto& w : solid_wavefield_writers_) w->Finalize();
}

// ---------------------------------------------------------------------------
// Observability configuration / accessors
// ---------------------------------------------------------------------------

template<int Dim>
int CoupledSimulationFacade<Dim>::LogInterval() const {
    return config_ ? config_->GetLogInterval() : 0;
}

template<int Dim>
std::string CoupledSimulationFacade<Dim>::OutputDir() const {
    return config_ ? config_->GetOutputDirectory() : std::string{};
}

template<int Dim>
bool CoupledSimulationFacade<Dim>::IsWavefieldOutputEnabled() const {
    return config_ ? config_->IsWavefieldOutputEnabled() : false;
}

template<int Dim>
int CoupledSimulationFacade<Dim>::WavefieldInterval() const {
    return config_ ? config_->GetWavefieldInterval() : 0;
}

template<int Dim>
std::string CoupledSimulationFacade<Dim>::SummaryFile() const {
    if (!config_) return std::string{};
    const std::string explicit_path = config_->GetSummaryFile();
    if (!explicit_path.empty()) return explicit_path;
    const std::string outdir = OutputDir();
    return outdir.empty() ? std::string{} : (outdir + "/summary.txt");
}

template<int Dim>
real_t CoupledSimulationFacade<Dim>::T0() const {
    return config_ ? config_->GetT0() : real_t(0);
}

template<int Dim>
CoupledSimulationFacade<Dim>&
CoupledSimulationFacade<Dim>::EnableProgressOutput(int interval)
{
    const std::string outdir = OutputDir();
    log_file_ = outdir.empty() ? std::string{} : outdir + "/log.txt";
    // Rank 0 truncates the shared log file once so both loggers append
    // cleanly afterwards.
    if (IsRoot() && !log_file_.empty()) {
        std::ofstream ofs(log_file_, std::ios::trunc);
    }
    fluid_progress_.Configure(interval, log_file_, "fluid", comm_, IsRoot());
    solid_progress_.Configure(interval, log_file_, "solid", comm_, IsRoot());
    return *this;
}

template<int Dim>
CoupledSimulationFacade<Dim>&
CoupledSimulationFacade<Dim>::SetFluidWavefieldWriters(
    std::vector<std::unique_ptr<WavefieldWriter>> writers)
{
    fluid_wavefield_writers_ = std::move(writers);
    return *this;
}

template<int Dim>
CoupledSimulationFacade<Dim>&
CoupledSimulationFacade<Dim>::SetSolidWavefieldWriters(
    std::vector<std::unique_ptr<WavefieldWriter>> writers)
{
    solid_wavefield_writers_ = std::move(writers);
    return *this;
}

template<int Dim>
bool CoupledSimulationFacade<Dim>::CheckCFL(real_t cfl_factor,
                                            bool   abort_on_violation)
{
    MFEM_VERIFY(setup_done_,
                "CheckCFL called before SetupFromConfig");
    const int fluid_order = fluid_comp_.Order();
    const int solid_order = solid_comp_.Order();
    const bool fluid_ok = CheckCFLOnSubmesh<Dim>(
        fluid_comp_.Mesh(), fluid_order, *fluid_material_,
        dt_, cfl_factor, comm_, "fluid CFL", abort_on_violation);
    const bool solid_ok = CheckCFLOnSubmesh<Dim>(
        solid_comp_.Mesh(), solid_order, *solid_material_,
        dt_, cfl_factor, comm_, "solid CFL", abort_on_violation);
    return fluid_ok && solid_ok;
}

template<int Dim>
bool CoupledSimulationFacade<Dim>::CheckWavelengthSampling(
    real_t f_max, real_t ppw_required, bool warn_on_insufficient)
{
    MFEM_VERIFY(setup_done_,
                "CheckWavelengthSampling called before SetupFromConfig");
    const bool fluid_ok = CheckWavelengthSamplingOnSubmesh<Dim>(
        fluid_comp_.Mesh(), fluid_comp_.Order(), *fluid_material_,
        f_max, ppw_required, comm_, "fluid", warn_on_insufficient);
    const bool solid_ok = CheckWavelengthSamplingOnSubmesh<Dim>(
        solid_comp_.Mesh(), solid_comp_.Order(), *solid_material_,
        f_max, ppw_required, comm_, "solid", warn_on_insufficient);
    return fluid_ok && solid_ok;
}

namespace {

/**
 * Per-domain (fluid or solid) summary ingredients that require MPI
 * collective reductions. Populated by BuildDomainStats on every rank;
 * consumed inside WriteSummaryToFile's rank-0 file writer.
 */
template<int Dim>
struct DomainSummaryStats {
    long long    global_ne    = 0;
    HYPRE_BigInt global_dofs  = 0;
    real_t       h_min        = 0.0;
    real_t       h_max        = 0.0;
    real_t       v_max        = 0.0;
    real_t       v_min        = 0.0;
    real_t       dt_cfl       = 0.0;
    real_t       ppw_achieved = 0.0;
};

template<int Dim>
DomainSummaryStats<Dim> BuildDomainStats(SimulationComponents<Dim>& comp,
                                         const MaterialBase&        mat,
                                         real_t                     cfl_factor,
                                         real_t                     f_source,
                                         MPI_Comm                   comm)
{
    DomainSummaryStats<Dim> s;
    s.global_ne   = comp.Mesh().GetGlobalNE();
    s.global_dofs = comp.FES().GlobalTrueVSize();

    // h_min / h_max across the submesh (matches pure path; type=1 uses
    // the inradius-based h, same metric SimulationFacade uses).
    real_t h_min_local = std::numeric_limits<real_t>::max();
    real_t h_max_local = 0.0;
    for (int e = 0; e < comp.Mesh().GetNE(); ++e) {
        const real_t h = comp.Mesh().GetElementSize(e, 1);
        if (h < h_min_local) h_min_local = h;
        if (h > h_max_local) h_max_local = h;
    }
    MPI_Allreduce(&h_min_local, &s.h_min, 1, MFEM_MPI_REAL_T, MPI_MIN, comm);
    MPI_Allreduce(&h_max_local, &s.h_max, 1, MFEM_MPI_REAL_T, MPI_MAX, comm);

    // v_max / v_min via the material (P/S extrema per element, reduced).
    real_t v_max_local = mat.GetMaxVelocity();
    real_t v_min_local = mat.GetMinVelocity();
    MPI_Allreduce(&v_max_local, &s.v_max, 1, MFEM_MPI_REAL_T, MPI_MAX, comm);
    MPI_Allreduce(&v_min_local, &s.v_min, 1, MFEM_MPI_REAL_T, MPI_MIN, comm);

    // Per-element CFL dt_max and PPW exactly as pure does (shared helper).
    const int N = comp.Order();
    real_t min_dt_local = std::numeric_limits<real_t>::max();
    for (int e = 0; e < comp.Mesh().GetNE(); ++e) {
        const real_t dist_min = ComputeElementMinGLLDistance<Dim>(comp.Mesh(), e, N);
        const real_t v_elem   = mat.GetElementMaxVelocity(e);
        const real_t dt_max_e = cfl_factor * dist_min / v_elem;
        if (dt_max_e < min_dt_local) min_dt_local = dt_max_e;
    }
    MPI_Allreduce(&min_dt_local, &s.dt_cfl, 1, MFEM_MPI_REAL_T, MPI_MIN, comm);

    if (f_source > 0) {
        const int ngll = N + 1;
        real_t min_ppw_local = std::numeric_limits<real_t>::max();
        for (int e = 0; e < comp.Mesh().GetNE(); ++e) {
            const real_t h      = comp.Mesh().GetElementSize(e, 2);
            const real_t v_elem = mat.GetElementMinVelocity(e);
            const real_t ppw_e  = v_elem * ngll / (f_source * h);
            if (ppw_e < min_ppw_local) min_ppw_local = ppw_e;
        }
        MPI_Allreduce(&min_ppw_local, &s.ppw_achieved, 1, MFEM_MPI_REAL_T,
                      MPI_MIN, comm);
    }
    return s;
}

/**
 * Emit one full per-domain block in the same layout pure's
 * WriteSummaryToFile uses for its single-domain case: Material,
 * Discretization, Operator, Numerical stability. Kept identical so
 * `diff <pure-summary> <coupled-fluid-section>` inside a single domain
 * is (modulo the `-- <side> domain --` banner) byte-for-byte the same.
 */
template<int Dim>
void WriteDomainSection(std::ostream&                     ofs,
                        const char*                       label,
                        const SimulationComponents<Dim>&  comp,
                        const MaterialBase&               mat,
                        WaveOperator*                     op,
                        const DomainSummaryStats<Dim>&    s,
                        real_t                            dt,
                        real_t                            f_source)
{
    ofs << "-- " << label << " domain --\n";

    ofs << "Material: " << MaterialTypeToString(mat.GetType()) << "\n";
    ofs << std::fixed << std::setprecision(2);
    ofs << "  Vmax: " << s.v_max << " m/s, Vmin: " << s.v_min << " m/s\n";
    if (mat.HasAttenuation()) {
        ofs << "  Attenuation: enabled (f0=" << mat.AttenuationF0()
            << " Hz, " << mat.AttenuationNumUnits() << " SLS units)\n";
    } else {
        ofs << "  Attenuation: disabled\n";
    }
    ofs << std::defaultfloat;

    ofs << "\nDiscretization:\n";
    ofs << "  Order: " << comp.Order() << "\n";
    if (s.global_ne   > 0) ofs << "  Elements: " << s.global_ne   << "\n";
    if (s.global_dofs > 0) ofs << "  DOFs: "     << s.global_dofs << "\n";

    if (op) {
        ofs << "\nOperator:\n";
        std::ostringstream op_oss;
        op->PrintInfo(op_oss);
        std::string line;
        std::istringstream op_iss(op_oss.str());
        while (std::getline(op_iss, line)) ofs << "  " << line << "\n";
    }

    ofs << "\nNumerical stability:\n";
    ofs << std::fixed << std::setprecision(2);
    ofs << "  h_min: " << s.h_min << " m, h_max: " << s.h_max << " m\n";
    ofs << "  CFL dt_max: " << std::scientific << std::setprecision(4) << s.dt_cfl
        << " s (dt/dt_max = " << std::fixed << std::setprecision(2)
        << (dt / s.dt_cfl) << ")\n";
    if (s.ppw_achieved > 0) {
        ofs << "  PPW: " << std::fixed << std::setprecision(1) << s.ppw_achieved
            << " (f=" << f_source << " Hz)\n";
    }
    ofs << std::defaultfloat;
}

}  // namespace

template<int Dim>
void CoupledSimulationFacade<Dim>::WriteSummaryToFile(
    const std::string& path, const TimingInfo& timing)
{
    // NOTE: every statistics call below is MPI-collective — do them on
    // all ranks before gating file I/O on IsRoot(). Layout mirrors
    // SimulationFacade<Dim>::WriteSummaryToFile so `diff <pure> <coupled>`
    // shows a fluid/solid pair of identical per-domain blocks instead of
    // a completely foreign format.

    // Pick the highest source frequency (shared between both domains, same
    // convention as pure). Used for PPW on each submesh.
    real_t f_source = 0.0;
    if (config_) {
        auto all_sources = config_->GetAllSources();
        for (const auto& src : all_sources) {
            if (src.frequency > f_source) f_source = src.frequency;
        }
    }
    const real_t cfl_factor = config_ ? config_->GetCflFactor() : real_t(0.3);

    const auto fluid_stats = BuildDomainStats<Dim>(
        fluid_comp_, *fluid_material_, cfl_factor, f_source, comm_);
    const auto solid_stats = BuildDomainStats<Dim>(
        solid_comp_, *solid_material_, cfl_factor, f_source, comm_);

    // Interface quadrature count (MPI sum).
    long long iface_quad_local  = iface_.NumLocalQuadPoints();
    long long iface_quad_global = 0;
    MPI_Allreduce(&iface_quad_local, &iface_quad_global, 1, MPI_LONG_LONG,
                  MPI_SUM, comm_);

    if (!IsRoot()) return;

    std::ofstream ofs(path);
    if (!ofs.is_open()) return;

    // ---- Header (matches pure) --------------------------------------
    if (config_) ofs << "# Simulation: " << config_->GetName() << "\n";
    ofs << "# Generated by SEMSWS\n\n";
    ofs << "========================================\n";
    ofs << "        SEMSWS " << Dim << "D Coupled Simulation\n";
    ofs << "========================================\n\n";

    // ---- Per-domain sections (same shape as pure, one per side) -----
    WriteDomainSection<Dim>(ofs, "Fluid", fluid_comp_, *fluid_material_,
                            fluid_op_.get(), fluid_stats, dt_, f_source);
    ofs << "\n";
    WriteDomainSection<Dim>(ofs, "Solid", solid_comp_, *solid_material_,
                            solid_op_.get(), solid_stats, dt_, f_source);

    // ---- Fluid-solid interface (coupled-only section) ---------------
    ofs << "\nInterface:\n";
    ofs << "  Attribute: "   << iface_.InterfaceAttribute() << "\n";
    ofs << "  Quad points: " << iface_quad_global << "\n";

    // ---- Combined source / receiver counts --------------------------
    const int n_src_total =
        (fluid_sources_   ? fluid_sources_->NumSources()     : 0) +
        (solid_sources_   ? solid_sources_->NumSources()     : 0);
    const int n_rcv_total =
        (fluid_receivers_ ? fluid_receivers_->NumReceivers() : 0) +
        (solid_receivers_ ? solid_receivers_->NumReceivers() : 0);
    if (n_src_total > 0) {
        ofs << "\nSources: " << n_src_total << " total ("
            << (fluid_sources_ ? fluid_sources_->NumSources() : 0) << " fluid, "
            << (solid_sources_ ? solid_sources_->NumSources() : 0) << " solid)\n";
    }
    if (n_rcv_total > 0) {
        ofs << "\nReceivers: " << n_rcv_total << " total ("
            << (fluid_receivers_ ? fluid_receivers_->NumReceivers() : 0) << " fluid, "
            << (solid_receivers_ ? solid_receivers_->NumReceivers() : 0) << " solid)\n";
    }

    // ---- Time stepping (shared between domains) ---------------------
    ofs << "\nTime stepping:\n";
    ofs << "  dt: " << dt_ << " s\n";
    ofs << "  nt: " << nt_ << "\n";
    ofs << "  Simulation time: " << (dt_ * nt_) << " s\n";

    // ---- Parallel configuration -------------------------------------
    ofs << "\nParallel configuration:\n";
    ofs << "  Device: " << (config_ ? config_->GetDevice() : "cpu") << "\n";
    ofs << "  MPI processes: " << num_procs_ << "\n";

    // ---- Performance metrics ----------------------------------------
    ofs << "\nPerformance:\n";
    ofs << std::fixed << std::setprecision(4);
    ofs << "  Setup time:  " << timing.setup_time << " s\n";
    ofs << "  Run time:    " << timing.run_time   << " s\n";
    ofs << "  IO time:     " << timing.io_time    << " s\n";
    ofs << "  Total time:  " << timing.total_time << " s\n";
    if (timing.run_time > 0) {
        // The "active" DOF count for the coupled run is the sum of both
        // domains (each timestep updates both), so report combined
        // throughput to stay comparable with the pure summary's DOFs/sec.
        const HYPRE_BigInt total_dofs = fluid_stats.global_dofs +
                                        solid_stats.global_dofs;
        ofs << "  Steps/sec:   " << std::fixed << std::setprecision(4)
            << (nt_ / timing.run_time) << "\n";
        ofs << "  Time/step:   " << std::fixed << std::setprecision(4)
            << (timing.run_time / nt_ * 1000.0) << " ms\n";
        const double dofs_per_sec =
            static_cast<double>(total_dofs) * nt_ / timing.run_time;
        ofs << "  DOFs/sec:    " << std::scientific << std::setprecision(2)
            << dofs_per_sec << "\n";
    }
    ofs << std::defaultfloat;

    ofs << "========================================\n";

    ofs.close();
}

template<int Dim>
void CoupledSimulationFacade<Dim>::SaveReceivers()
{
    const real_t t0 = config_ ? config_->GetT0() : real_t(0);

    // Mirror ForwardSimulation::Run — first source on whichever side owns
    // one provides the id / position used as the filename + SU header
    // metadata for BOTH receiver arrays. All sources fire simultaneously
    // within the single shot, so the pair (id, position) is unambiguous.
    int id = 0;
    const Vector* pos = nullptr;
    if (solid_sources_ && solid_sources_->NumSources() > 0) {
        id  = solid_sources_->GetSource(0)->GetId();
        pos = &solid_sources_->GetSource(0)->Position();
    } else if (fluid_sources_ && fluid_sources_->NumSources() > 0) {
        id  = fluid_sources_->GetSource(0)->GetId();
        pos = &fluid_sources_->GetSource(0)->Position();
    }

    if (fluid_receivers_) fluid_receivers_->Save(id, t0, pos);
    if (solid_receivers_) solid_receivers_->Save(id, t0, pos);
}

// ---------------------------------------------------------------------------
// Explicit instantiation
// ---------------------------------------------------------------------------

template class CoupledSimulationFacade<2>;
template class CoupledSimulationFacade<3>;

}  // namespace SEM
