/**
 * @file SensitivityKernelFactory.hpp
 * @brief Factory functions that construct the FWI sensitivity kernel matched
 *        to a given (domain, dimension, material) combination.
 *
 * Mirrors the pattern used by integ/StiffnessIntegratorFactory.hpp: four
 * narrow entry points, one per (physics, dim) pair. The caller (currently
 * AdjointSimulation) already knows the dimension from its template parameter
 * and the domain from the material, so each factory only needs to dispatch
 * on the concrete material type.
 *
 * Only 2D acoustic is implemented today (IsotropicAcousticSensitivity2D).
 * The other three entry points exist so AdjointSimulation can route calls
 * uniformly; they MFEM_ABORT until a concrete kernel is added.
 */

#ifndef SEM_SENSITIVITY_KERNEL_FACTORY_HPP
#define SEM_SENSITIVITY_KERNEL_FACTORY_HPP

#include <mfem.hpp>
#include <memory>
#include <string>

namespace SEM {

class ElasticMaterialBase2D;
class ElasticMaterialBase3D;
class AcousticMaterialBase2D;
class AcousticMaterialBase3D;
class SensitivityKernelBase2D;
class SensitivityKernelBase3D;

/// Build the 2D elastic FWI sensitivity kernel.
///
/// @param backend Selector for the implementation backend:
///   - "hand" (default): IsotropicElasticSensitivity2D with a classical
///     chain-rule accumulation; writes K_Vp, K_Vs, K_ρ.
///   - "ad": reserved for the forward-mode AD version (not yet landed).
std::unique_ptr<SensitivityKernelBase2D>
CreateElasticSensitivityKernel2D(const ElasticMaterialBase2D& material,
                                 mfem::ParFiniteElementSpace& fes,
                                 const std::string& backend = "hand");

/// Build the 3D elastic FWI sensitivity kernel.
///
/// @param backend "hand" (IsotropicElasticSensitivity3D, classical chain rule)
///                or "ad" (IsotropicElasticSensitivityAD3D, forward-mode AD).
///                Both write K_Vp, K_Vs, K_ρ.
std::unique_ptr<SensitivityKernelBase3D>
CreateElasticSensitivityKernel3D(const ElasticMaterialBase3D& material,
                                 mfem::ParFiniteElementSpace& fes,
                                 const std::string& backend = "hand");

/// Build the 2D acoustic FWI sensitivity kernel.
/// Currently returns IsotropicAcousticSensitivity2D for IsotropicAcoustic;
/// aborts for any other acoustic material type.
///
/// @param backend Selector for the implementation backend:
///   - "hand" : hand-derived chain-rule implementation (default, production)
///   - "ad"   : forward-mode automatic differentiation via mfem::future::dual
///              (same outputs to numerical tolerance, see
///              test/unit/acoustic_ad_vs_hand_test.cpp for verification)
std::unique_ptr<SensitivityKernelBase2D>
CreateAcousticSensitivityKernel2D(const AcousticMaterialBase2D& material,
                                  mfem::ParFiniteElementSpace& fes,
                                  const std::string& backend = "hand");

/// Build the 3D acoustic FWI sensitivity kernel.
///
/// @param backend "hand" (IsotropicAcousticSensitivity3D) or "ad"
///                (IsotropicAcousticSensitivityAD3D).
std::unique_ptr<SensitivityKernelBase3D>
CreateAcousticSensitivityKernel3D(const AcousticMaterialBase3D& material,
                                  mfem::ParFiniteElementSpace& fes,
                                  const std::string& backend = "hand");

}  // namespace SEM

#endif  // SEM_SENSITIVITY_KERNEL_FACTORY_HPP
