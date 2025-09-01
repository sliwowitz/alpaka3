// SPDX-License-Identifier: Apache-2.0
// Portions adapted conceptually from UoB-HPC/miniBUDE (Apache-2.0).
// Author: Ivan Andriievskyi
// Work funded by US NAS and ONRG (IMPRESS-U).

#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// ----------------------------
// Minimal reference "physics"
// ----------------------------
//
// CPU-only, serial implementation of the core scoring for a single pose:
//   E_total = Lennard–Jones(12-6) + Coulomb
// This file implements LJ and Coulomb terms; other terms can be incorporated separately if needed.
//
// Units:
// * We keep k_e consistent with kJ·nm/(mol·e^2) for now (synthetic deck).
// * With a real deck, units may be adjusted (e.g., kcal·Å) as required.

// Atom parameters (per-particle)
struct Atom
{
    float x, y, z; // position
    float q; // partial charge (e)
    float sigma; // LJ sigma (Å or nm — consistent within the deck)
    float epsilon; // LJ epsilon (kcal/mol or kJ/mol — consistent within the deck)
};

// Rigid-body pose: pure translation (rotations are not included here)
struct Pose
{
    float tx, ty, tz;
};

// Read-only view on the system arrays
struct SystemView
{
    Atom const* ligand;
    std::size_t natlig;
    Atom const* protein;
    std::size_t natpro;
};

// Coulomb constant (kJ·nm)/(mol·e^2). For kcal·Å you would typically use ~332.06371f.
inline constexpr float k_e = 138.935456f;

// Pairwise LJ(12-6) + Coulomb for distance-squared r2 and mixed (sigma, epsilon).
inline float pair_energy(float r2, float q1, float q2, float sigma, float epsilon)
{
    // Avoid r = 0 under synthetic inputs
    r2 = std::max(r2, 1e-12f);
    float inv_r = 1.0f / std::sqrt(r2);

    // (sigma / r)
    float sr = sigma * inv_r;
    float sr2 = sr * sr;
    float sr6 = sr2 * sr2 * sr2; // (sigma/r)^6
    float sr12 = sr6 * sr6; // (sigma/r)^12

    // Lennard–Jones 12-6: 4*epsilon * [(sigma/r)^12 - (sigma/r)^6]
    float e_lj = 4.0f * epsilon * (sr12 - sr6);

    // Coulomb: k_e * q1*q2 / r
    float e_coul = k_e * q1 * q2 * inv_r;

    return e_lj + e_coul;
}

// Score a single pose: translate ligand, then accumulate all pair interactions.
inline float score_pose(SystemView const& sys, Pose const& pose)
{
    float E = 0.0f;

    for(std::size_t i = 0; i < sys.natlig; ++i)
    {
        Atom const& L = sys.ligand[i];
        // Apply translation (rotations will arrive with real deck)
        float const lx = L.x + pose.tx;
        float const ly = L.y + pose.ty;
        float const lz = L.z + pose.tz;

        for(std::size_t j = 0; j < sys.natpro; ++j)
        {
            Atom const& P = sys.protein[j];
            float const dx = lx - P.x;
            float const dy = ly - P.y;
            float const dz = lz - P.z;
            float const r2 = dx * dx + dy * dy + dz * dz;

            // Lorentz–Berthelot mixing:
            //   sigma_ij = (sigma_i + sigma_j)/2
            //   epsilon_ij = sqrt(epsilon_i * epsilon_j)
            float const sigma = 0.5f * (L.sigma + P.sigma);
            float const epsilon = std::sqrt(L.epsilon * P.epsilon);

            E += pair_energy(r2, L.q, P.q, sigma, epsilon);
        }
    }
    return E;
}
