// SPDX-License-Identifier: Apache-2.0
// Portions adapted from UoB-HPC/miniBUDE (Apache-2.0) serial backend.
// Author: Ivan Andriievskyi
// Work funded by US NAS and ONRG (IMPRESS-U).

#pragma once
#include "minibude.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

// ===== Alpaka kernel and host wrapper to compute energies for all poses =====

struct ScorePosesKernel1D
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto const ligand,
        alpaka::concepts::MdSpan auto const protein,
        alpaka::concepts::MdSpan auto const ff,
        alpaka::concepts::MdSpan auto const rx,
        alpaka::concepts::MdSpan auto const ry,
        alpaka::concepts::MdSpan auto const rz,
        alpaka::concepts::MdSpan auto const tx,
        alpaka::concepts::MdSpan auto const ty,
        alpaka::concepts::MdSpan auto const tz,
        alpaka::concepts::MdSpan auto outE,
        std::uint32_t natlig,
        std::uint32_t natpro,
        std::uint32_t nposes) const
    {
        auto threadIdxInGrid = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);
        auto numThreadsInGrid = acc.getExtentsOf(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);

        auto linearGridThreadIndex = alpaka::linearize(numThreadsInGrid, threadIdxInGrid);
        auto linearGridSize = numThreadsInGrid.product();

        for(std::uint32_t i = linearGridThreadIndex; i < nposes; i += linearGridSize)
        {
            float const sx = alpaka::math::sin(rx[i]), cx = alpaka::math::cos(rx[i]);
            float const sy = alpaka::math::sin(ry[i]), cy = alpaka::math::cos(ry[i]);
            float const sz = alpaka::math::sin(rz[i]), cz = alpaka::math::cos(rz[i]);

            float T[3][4];
            T[0][0] = cy * cz;
            T[0][1] = sx * sy * cz - cx * sz;
            T[0][2] = cx * sy * cz + sx * sz;
            T[0][3] = tx[i];

            T[1][0] = cy * sz;
            T[1][1] = sx * sy * sz + cx * cz;
            T[1][2] = cx * sy * sz - sx * cz;
            T[1][3] = ty[i];

            T[2][0] = -sy;
            T[2][1] = sx * cy;
            T[2][2] = cx * cy;
            T[2][3] = tz[i];

            float etot = 0.0f;

            for(std::uint32_t il = 0; il < natlig; ++il)
            {
                DeckAtom const l_atom = ligand[il];
                FFParams const l_params = ff[static_cast<std::uint32_t>(l_atom.type)];
                bool const lhphb_ltz = (l_params.hphb < 0.f);
                bool const lhphb_gtz = (l_params.hphb > 0.f);

                float const lpos_x = T[0][3] + l_atom.x * T[0][0] + l_atom.y * T[0][1] + l_atom.z * T[0][2];
                float const lpos_y = T[1][3] + l_atom.x * T[1][0] + l_atom.y * T[1][1] + l_atom.z * T[1][2];
                float const lpos_z = T[2][3] + l_atom.x * T[2][0] + l_atom.y * T[2][1] + l_atom.z * T[2][2];

                for(std::uint32_t ip = 0; ip < natpro; ++ip)
                {
                    DeckAtom const p_atom = protein[ip];
                    FFParams const p_params = ff[static_cast<std::uint32_t>(p_atom.type)];

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

                    float const distdslv
                        = (phphb_ltz ? (lhphb_ltz ? NPNPDIST : NPPDIST) : (lhphb_ltz ? NPPDIST : -FloatMax));
                    float const r_distdslv = (distdslv != 0.f) ? (ONE / distdslv) : 0.f;

                    float const chrg_init = l_params.elsc * p_params.elsc;
                    float const dslv_init = p_hphb + l_hphb;

                    float const dx = lpos_x - p_atom.x;
                    float const dy = lpos_y - p_atom.y;
                    float const dz = lpos_z - p_atom.z;
                    float const distij = alpaka::math::sqrt(dx * dx + dy * dy + dz * dz);

                    float const distbb = distij - radij;
                    bool const zone1 = (distbb < ZERO);

                    etot += (ONE - (distij * r_radij)) * (zone1 ? TWO * HARDNESS : 0.f);

                    float chrg_e
                        = chrg_init * ((zone1 ? ONE : (ONE - distbb * elcdst1)) * ((distbb < elcdst) ? ONE : ZERO));
                    if(type_E)
                        chrg_e = -alpaka::math::abs(chrg_e);
                    etot += chrg_e * CNSTNT;

                    float const coeff = (ONE - (distbb * r_distdslv));
                    float dslv_e = dslv_init * (((distbb < distdslv) && phphb_nz) ? ONE : 0.f);
                    dslv_e *= (zone1 ? ONE : coeff);
                    etot += dslv_e;
                }
            }

            outE[i] = etot * HALF;
        }
    }
};

inline std::vector<float> compute_energies_alpaka(
    std::vector<DeckAtom> const& ligand,
    std::vector<DeckAtom> const& protein,
    std::vector<FFParams> const& ff,
    std::array<std::vector<float>, 6> const& dof)
{
    using namespace alpaka;
    auto const N = static_cast<std::uint32_t>(dof[0].size());
    auto const natlig = static_cast<std::uint32_t>(ligand.size());
    auto const natpro = static_cast<std::uint32_t>(protein.size());

    std::vector<float> energies(N, 0.0f);

    bool executed = false;
    (void) alpaka::executeForEachIfHasDevice(
        [&](auto const& backend)
        {
            if(executed)
                return;

            auto devSelector = onHost::makeDeviceSelector(backend[object::deviceSpec]);
            if(!devSelector.isAvailable())
                return;

            onHost::Device dev = devSelector.makeDevice(0);
            onHost::Queue queue = dev.makeQueue();

            // Host-pinned views and device mirrors
            auto ligand_h = onHost::allocHost<DeckAtom>(Vec<uint32_t, 1u>{natlig});
            auto protein_h = onHost::allocHost<DeckAtom>(Vec<uint32_t, 1u>{natpro});
            auto ff_h = onHost::allocHost<FFParams>(Vec<uint32_t, 1u>{static_cast<std::uint32_t>(ff.size())});
            auto rx_h = onHost::allocHost<float>(Vec<uint32_t, 1u>{N});
            auto ry_h = onHost::allocHost<float>(Vec<uint32_t, 1u>{N});
            auto rz_h = onHost::allocHost<float>(Vec<uint32_t, 1u>{N});
            auto tx_h = onHost::allocHost<float>(Vec<uint32_t, 1u>{N});
            auto ty_h = onHost::allocHost<float>(Vec<uint32_t, 1u>{N});
            auto tz_h = onHost::allocHost<float>(Vec<uint32_t, 1u>{N});
            auto out_h = onHost::allocHost<float>(Vec<uint32_t, 1u>{N});

            // Copy from std::vector to host views
            for(std::uint32_t i = 0; i < natlig; ++i)
                ligand_h[i] = ligand[i];
            for(std::uint32_t i = 0; i < natpro; ++i)
                protein_h[i] = protein[i];
            for(std::uint32_t i = 0; i < static_cast<std::uint32_t>(ff.size()); ++i)
                ff_h[i] = ff[i];
            for(std::uint32_t i = 0; i < N; ++i)
            {
                rx_h[i] = dof[0][i];
                ry_h[i] = dof[1][i];
                rz_h[i] = dof[2][i];
                tx_h[i] = dof[3][i];
                ty_h[i] = dof[4][i];
                tz_h[i] = dof[5][i];
                out_h[i] = 0.0f;
            }

            auto ligand_d = onHost::allocMirror(dev, ligand_h);
            auto protein_d = onHost::allocMirror(dev, protein_h);
            auto ff_d = onHost::allocMirror(dev, ff_h);
            auto rx_d = onHost::allocMirror(dev, rx_h);
            auto ry_d = onHost::allocMirror(dev, ry_h);
            auto rz_d = onHost::allocMirror(dev, rz_h);
            auto tx_d = onHost::allocMirror(dev, tx_h);
            auto ty_d = onHost::allocMirror(dev, ty_h);
            auto tz_d = onHost::allocMirror(dev, tz_h);
            auto out_d = onHost::allocMirror(dev, out_h);

            onHost::memcpy(queue, ligand_d, ligand_h);
            onHost::memcpy(queue, protein_d, protein_h);
            onHost::memcpy(queue, ff_d, ff_h);
            onHost::memcpy(queue, rx_d, rx_h);
            onHost::memcpy(queue, ry_d, ry_h);
            onHost::memcpy(queue, rz_d, rz_h);
            onHost::memcpy(queue, tx_d, tx_h);
            onHost::memcpy(queue, ty_d, ty_h);
            onHost::memcpy(queue, tz_d, tz_h);
            onHost::memset(queue, out_d, 0x00);

            // Thread spec: choose a sensible 1D configuration
            auto computeExec = backend[object::exec];
            std::uint32_t const threadsPerBlock = 256u;
            auto threadSpec = onHost::ThreadSpec{(N + threadsPerBlock - 1u) / threadsPerBlock, threadsPerBlock};
            if constexpr(alpaka::isSeqExecutor(ALPAKA_TYPEOF(computeExec){}))
            {
                // Force single thread per block for seq executors
                threadSpec.m_numThreads = decltype(threadSpec.m_numThreads){1u};
                threadSpec.m_numBlocks = decltype(threadSpec.m_numBlocks){N};
            }

            queue.enqueue(
                computeExec,
                threadSpec,
                ScorePosesKernel1D{},
                ligand_d,
                protein_d,
                ff_d,
                rx_d,
                ry_d,
                rz_d,
                tx_d,
                ty_d,
                tz_d,
                out_d,
                natlig,
                natpro,
                N);

            onHost::memcpy(queue, out_h, out_d);
            onHost::wait(queue);

            for(std::uint32_t i = 0; i < N; ++i)
                energies[i] = out_h[i];

            executed = true;
        },
        onHost::allBackends(onHost::enabledApis));

    return energies;
}
