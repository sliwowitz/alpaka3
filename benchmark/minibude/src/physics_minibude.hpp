// SPDX-License-Identifier: Apache-2.0
// Portions adapted from UoB-HPC/miniBUDE (Apache-2.0) serial backend.
// Author: Ivan Andriievskyi
// Work funded by US NAS and ONRG (IMPRESS-U).

#pragma once
#include "minibude.hpp"

#include <cmath>
#include <cstdint>

inline float score_pose_minibude_serial(
    DeckAtom const* ligand,
    std::size_t natlig,
    DeckAtom const* protein,
    std::size_t natpro,
    FFParams const* ff,
    float rx,
    float ry,
    float rz,
    float tx,
    float ty,
    float tz)
{
    float const sx = std::sin(rx), cx = std::cos(rx);
    float const sy = std::sin(ry), cy = std::cos(ry);
    float const sz = std::sin(rz), cz = std::cos(rz);

    float T[3][4];
    T[0][0] = cy * cz;
    T[0][1] = sx * sy * cz - cx * sz;
    T[0][2] = cx * sy * cz + sx * sz;
    T[0][3] = tx;

    T[1][0] = cy * sz;
    T[1][1] = sx * sy * sz + cx * cz;
    T[1][2] = cx * sy * sz - sx * cz;
    T[1][3] = ty;

    T[2][0] = -sy;
    T[2][1] = sx * cy;
    T[2][2] = cx * cy;
    T[2][3] = tz;

    float etot = 0.0f;

    for(std::size_t il = 0; il < natlig; ++il)
    {
        DeckAtom const l_atom = ligand[il];
        FFParams const l_params = ff[static_cast<std::size_t>(l_atom.type)];
        bool const lhphb_ltz = (l_params.hphb < 0.f);
        bool const lhphb_gtz = (l_params.hphb > 0.f);

        float const lpos_x = T[0][3] + l_atom.x * T[0][0] + l_atom.y * T[0][1] + l_atom.z * T[0][2];
        float const lpos_y = T[1][3] + l_atom.x * T[1][0] + l_atom.y * T[1][1] + l_atom.z * T[1][2];
        float const lpos_z = T[2][3] + l_atom.x * T[2][0] + l_atom.y * T[2][1] + l_atom.z * T[2][2];

        for(std::size_t ip = 0; ip < natpro; ++ip)
        {
            DeckAtom const p_atom = protein[ip];
            FFParams const p_params = ff[static_cast<std::size_t>(p_atom.type)];

            float const radij = p_params.radius + l_params.radius;
            float const r_radij = ONE / radij;

            bool const bothF = (p_params.hbtype == HBTYPE_F && l_params.hbtype == HBTYPE_F);
            float const elcdst = bothF ? FOUR : TWO;
            float const elcdst1 = bothF ? QUARTER : HALF;
            bool const type_E = (p_params.hbtype == HBTYPE_E || l_params.hbtype == HBTYPE_E);

            bool const phphb_ltz = (p_params.hphb < 0.f);
            bool const phphb_gtz = (p_params.hphb > 0.f);
            bool const phphb_nz = (p_params.hphb != 0.f);

            float const p_hphb = p_params.hphb * ((phphb_ltz && lhphb_gtz) ? -ONE : ONE);
            float const l_hphb = l_params.hphb * ((phphb_gtz && lhphb_ltz) ? -ONE : ONE);

            float const distdslv = (phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST) : (lhphb_ltz ? NPPDIST : -FloatMax));
            float const r_distdslv = (distdslv != 0.f) ? (ONE / distdslv) : 0.f;

            float const chrg_init = l_params.elsc * p_params.elsc;
            float const dslv_init = p_hphb + l_hphb;

            float const dx = lpos_x - p_atom.x;
            float const dy = lpos_y - p_atom.y;
            float const dz = lpos_z - p_atom.z;
            float const distij = std::sqrt(dx * dx + dy * dy + dz * dz);

            float const distbb = distij - radij;
            bool const zone1 = (distbb < ZERO);

            etot += (ONE - (distij * r_radij)) * (zone1 ? TWO * HARDNESS : 0.f);

            float chrg_e = chrg_init * ((zone1 ? ONE : (ONE - distbb * elcdst1)) * ((distbb < elcdst) ? ONE : ZERO));
            if(type_E)
                chrg_e = -std::fabs(chrg_e);
            etot += chrg_e * CNSTNT;

            float const coeff = (ONE - (distbb * r_distdslv));
            float dslv_e = dslv_init * (((distbb < distdslv) && phphb_nz) ? ONE : 0.f);
            dslv_e *= (zone1 ? ONE : coeff);
            etot += dslv_e;
        }
    }

    return etot * HALF;
}
