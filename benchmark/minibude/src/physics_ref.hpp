// SPDX-License-Identifier: Apache-2.0
// Portions adapted conceptually from UoB-HPC/miniBUDE (Apache-2.0).
// Author: Ivan Andriievskyi
// Work funded by US NAS and ONRG (IMPRESS-U).

#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

// ----------------------------
// Minimal reference "physics"
// ----------------------------
//
// This is a CPU-only, serial implementation of the core scoring:
//   E_total = Lennard-Jones + Coulomb
// (Additional terms can be incorporated later alongside real deck I/O and validation.)
//
// Notes:
// * Coordinates are in arbitrary units for now (synthetic deck).
// * Some constants are placeholders; align with the miniBUDE reference when wiring the real deck.

struct Atom
{
    float x, y, z; // position
    float q; // charge (e)
    float sigma; // LJ sigma (Å)
    float epsilon; // LJ epsilon (kcal/mol)
};

struct Pose
{
    float tx, ty, tz;
};

struct SystemView
{
    Atom const* ligand;
    std::size_t natlig;
    Atom const* protein;
    std::size_t natpro;
};

// Coulomb constant
inline constexpr float k_e = 138.935456f; // (kJ·nm)/(mol·e^2)

// Pairwise LJ + Coulomb between two atoms at distance r.
inline float pair_energy(float r2, float q1, float q2, float sigma, float epsilon)
{
    // Guard against r=0 in synthetic data
    r2 = std::max(r2, 1e-12f);
    float inv_r = 1.0f / std::sqrt(r2);
    float sr = sigma * inv_r;
    float sr2 = sr * sr;
    float sr6 = sr2 * sr2 * sr2;
    float sr12 = sr6 * sr6;

    // Lennard–Jones 12-6
    float e_lj = 4.0f * epsilon * (sr12 - sr6);

    // Coulomb
    float e_coul = k_e * q1 * q2 * inv_r;

    return e_lj + e_coul;
}

// Score one pose: translate ligand and accumulate pairwise interactions.
inline float score_pose(SystemView const& sys, Pose const& pose)
{
    float E = 0.0f;

    for(std::size_t i = 0; i < sys.natlig; ++i)
    {
        Atom const& L = sys.ligand[i];
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

            // Simple mixing for (sigma, epsilon): Lorentz–Berthelot
            float const sigma = 0.5f * (L.sigma + P.sigma);
            float const epsilon = std::sqrt(L.epsilon * P.epsilon);

            E += pair_energy(r2, L.q, P.q, sigma, epsilon);
        }
    }
    return E;
}
