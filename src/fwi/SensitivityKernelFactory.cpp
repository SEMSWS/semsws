/**
 * @file SensitivityKernelFactory.cpp
 * @brief Implementation of the four sensitivity-kernel factory entry points.
 *
 * Current coverage:
 *   - 2D Acoustic / IsotropicAcoustic → IsotropicAcousticSensitivity2D
 *
 * All other combinations abort with a descriptive message. Add new cases
 * here as elastic / 3D / anisotropic sensitivity kernels land.
 */

#include "fwi/SensitivityKernelFactory.hpp"
#include "fwi/SensitivityKernel.hpp"
#include "fwi/IsotropicAcousticSensitivity2D.hpp"
#include "fwi/IsotropicAcousticSensitivityAD2D.hpp"
#include "fwi/IsotropicAcousticSensitivity3D.hpp"
#include "fwi/IsotropicAcousticSensitivityAD3D.hpp"
#include "fwi/IsotropicElasticSensitivity2D.hpp"
#include "fwi/IsotropicElasticSensitivityAD2D.hpp"
#include "fwi/IsotropicElasticSensitivity3D.hpp"
#include "fwi/IsotropicElasticSensitivityAD3D.hpp"
#include "material/ElasticMaterialBase.hpp"
#include "material/AcousticMaterialBase.hpp"
#include "material/isotropic_acoustic/IsotropicAcousticMaterial.hpp"
#include "material/isotropic_elastic/IsotropicElasticMaterial.hpp"
#include "common/Types.hpp"

namespace SEM {

// =============================================================================
// 2D Elastic
// =============================================================================

std::unique_ptr<SensitivityKernelBase2D>
CreateElasticSensitivityKernel2D(const ElasticMaterialBase2D& material,
                                 mfem::ParFiniteElementSpace& fes,
                                 const std::string& backend)
{
    switch (material.GetType()) {
        case MaterialType::IsotropicElastic: {
            const auto& m = static_cast<const IsotropicElasticMaterial&>(material);
            const MaterialField* ck = m.UnrelaxedCorrectionKappa();
            const MaterialField* cm = m.UnrelaxedCorrectionMu();
            if (backend == "hand") {
                return std::make_unique<IsotropicElasticSensitivity2D>(
                    fes, m.Lambda(), m.Mu(), m.Rho(), ck, cm);
            } else if (backend == "ad") {
                return std::make_unique<IsotropicElasticSensitivityAD2D>(
                    fes, m.Lambda(), m.Mu(), m.Rho(), ck, cm);
            }
            MFEM_ABORT("Unknown inversion.sensitivity.backend: '" << backend
                       << "' (valid: hand, ad)");
            return nullptr;  // unreachable
        }
        default:
            MFEM_ABORT("Elastic 2D FWI sensitivity kernel not implemented for material type "
                       << MaterialTypeToString(material.GetType()));
            return nullptr;  // unreachable
    }
}

// =============================================================================
// 3D Elastic
// =============================================================================

std::unique_ptr<SensitivityKernelBase3D>
CreateElasticSensitivityKernel3D(const ElasticMaterialBase3D& material,
                                 mfem::ParFiniteElementSpace& fes,
                                 const std::string& backend)
{
    switch (material.GetType()) {
        case MaterialType::IsotropicElastic: {
            const auto& m = static_cast<const IsotropicElasticMaterial3D&>(material);
            const MaterialField3D* ck = m.UnrelaxedCorrectionKappa();
            const MaterialField3D* cm = m.UnrelaxedCorrectionMu();
            if (backend == "hand") {
                return std::make_unique<IsotropicElasticSensitivity3D>(
                    fes, m.Lambda(), m.Mu(), m.Rho(), ck, cm);
            } else if (backend == "ad") {
                return std::make_unique<IsotropicElasticSensitivityAD3D>(
                    fes, m.Lambda(), m.Mu(), m.Rho(), ck, cm);
            }
            MFEM_ABORT("Unknown inversion.sensitivity.backend: '" << backend
                       << "' (valid: hand, ad)");
            return nullptr;  // unreachable
        }
        default:
            MFEM_ABORT("Elastic 3D FWI sensitivity kernel not implemented for material type "
                       << MaterialTypeToString(material.GetType()));
            return nullptr;  // unreachable
    }
}

// =============================================================================
// 2D Acoustic
// =============================================================================

std::unique_ptr<SensitivityKernelBase2D>
CreateAcousticSensitivityKernel2D(const AcousticMaterialBase2D& material,
                                  mfem::ParFiniteElementSpace& fes,
                                  const std::string& backend)
{
    switch (material.GetType()) {
        case MaterialType::IsotropicAcoustic: {
            const auto& m = static_cast<const IsotropicAcousticMaterial&>(material);
            const MaterialField* corr = m.UnrelaxedCorrection();
            if (backend == "hand") {
                return std::make_unique<IsotropicAcousticSensitivity2D>(
                    fes, m.Kappa(), m.InvRho(), corr);
            } else if (backend == "ad") {
                return std::make_unique<IsotropicAcousticSensitivityAD2D>(
                    fes, m.Kappa(), m.InvRho(), corr);
            } else {
                MFEM_ABORT("Unknown inversion.sensitivity.backend: '" << backend
                           << "' (valid: hand, ad)");
                return nullptr;  // unreachable
            }
        }
        default:
            MFEM_ABORT("Acoustic 2D FWI sensitivity kernel not implemented for material type "
                       << MaterialTypeToString(material.GetType()));
            return nullptr;  // unreachable
    }
}

// =============================================================================
// 3D Acoustic
// =============================================================================

std::unique_ptr<SensitivityKernelBase3D>
CreateAcousticSensitivityKernel3D(const AcousticMaterialBase3D& material,
                                  mfem::ParFiniteElementSpace& fes,
                                  const std::string& backend)
{
    switch (material.GetType()) {
        case MaterialType::IsotropicAcoustic: {
            const auto& m = static_cast<const IsotropicAcousticMaterial3D&>(material);
            const MaterialField3D* corr = m.UnrelaxedCorrection();
            if (backend == "hand") {
                return std::make_unique<IsotropicAcousticSensitivity3D>(
                    fes, m.Kappa(), m.InvRho(), corr);
            } else if (backend == "ad") {
                return std::make_unique<IsotropicAcousticSensitivityAD3D>(
                    fes, m.Kappa(), m.InvRho(), corr);
            }
            MFEM_ABORT("Unknown inversion.sensitivity.backend: '" << backend
                       << "' (valid: hand, ad)");
            return nullptr;  // unreachable
        }
        default:
            MFEM_ABORT("Acoustic 3D FWI sensitivity kernel not implemented for material type "
                       << MaterialTypeToString(material.GetType()));
            return nullptr;  // unreachable
    }
}

}  // namespace SEM
