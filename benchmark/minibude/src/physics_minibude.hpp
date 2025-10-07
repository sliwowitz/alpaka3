// SPDX-License-Identifier: MPL-2.0
// Portions adapted from UoB-HPC/miniBUDE (Apache-2.0) serial backend.
// Author: Ivan Andriievskyi, Jiří Vyskočil
// Work funded by US NAS and ONRG (IMPRESS-U).

#pragma once
#include "minibude.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

namespace onHost = alpaka::onHost;
using alpaka::Vec;

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

template<typename TBackend>
struct MiniBudeContext
{
    using Backend = TBackend;
    using Clock = std::chrono::steady_clock;
    using Vec1U = Vec<uint32_t, 1u>;
    using HostDeckBuffer = std::decay_t<decltype(onHost::allocHost<DeckAtom>(Vec1U{1u}))>;
    using HostFFBuffer = std::decay_t<decltype(onHost::allocHost<FFParams>(Vec1U{1u}))>;
    using HostFloatBuffer = std::decay_t<decltype(onHost::allocHost<float>(Vec1U{1u}))>;
    using DeviceSpec = std::decay_t<decltype(std::declval<Backend const&>()[alpaka::object::deviceSpec])>;
    using ExecType = std::decay_t<decltype(std::declval<Backend const&>()[alpaka::object::exec])>;
    using DeviceHandle = std::decay_t<decltype(onHost::makeDeviceSelector(std::declval<DeviceSpec>()).makeDevice(0))>;
    using QueueHandle = std::decay_t<decltype(std::declval<DeviceHandle>().makeQueue())>;
    using ThreadSpecType = std::decay_t<decltype(onHost::ThreadSpec{Vec1U{1u}, Vec1U{1u}})>;
    using DeviceDeckBuffer = std::decay_t<
        decltype(onHost::allocLike(std::declval<DeviceHandle const&>(), std::declval<HostDeckBuffer const&>()))>;
    using DeviceFFBuffer = std::decay_t<
        decltype(onHost::allocLike(std::declval<DeviceHandle const&>(), std::declval<HostFFBuffer const&>()))>;
    using DeviceFloatBuffer = std::decay_t<
        decltype(onHost::allocLike(std::declval<DeviceHandle const&>(), std::declval<HostFloatBuffer const&>()))>;

    struct RunTimings
    {
        double kernel_ms;
        double d2h_ms;
    };

    Backend backend;
    DeviceHandle device;
    QueueHandle queue;
    ExecType exec;
    std::uint32_t nposes;
    std::uint32_t natlig;
    std::uint32_t natpro;
    std::uint32_t ffCount;
    ThreadSpecType threadSpec;

    HostDeckBuffer ligand_h;
    HostDeckBuffer protein_h;
    HostFFBuffer ff_h;
    HostFloatBuffer rx_h;
    HostFloatBuffer ry_h;
    HostFloatBuffer rz_h;
    HostFloatBuffer tx_h;
    HostFloatBuffer ty_h;
    HostFloatBuffer tz_h;
    HostFloatBuffer out_h;

    DeviceDeckBuffer ligand_d;
    DeviceDeckBuffer protein_d;
    DeviceFFBuffer ff_d;
    DeviceFloatBuffer rx_d;
    DeviceFloatBuffer ry_d;
    DeviceFloatBuffer rz_d;
    DeviceFloatBuffer tx_d;
    DeviceFloatBuffer ty_d;
    DeviceFloatBuffer tz_d;
    DeviceFloatBuffer out_d;

    MiniBudeContext(
        Backend const& backendIn,
        std::vector<DeckAtom> const& ligand,
        std::vector<DeckAtom> const& protein,
        std::vector<FFParams> const& ff,
        std::array<std::vector<float>, 6> const& dof)
        : backend(backendIn)
        , device(makeDevice(backendIn))
        , queue(device.makeQueue())
        , exec(backend[alpaka::object::exec])
        , nposes(static_cast<std::uint32_t>(dof[0].size()))
        , natlig(static_cast<std::uint32_t>(ligand.size()))
        , natpro(static_cast<std::uint32_t>(protein.size()))
        , ffCount(static_cast<std::uint32_t>(ff.size()))
        , ligand_h(onHost::allocHost<DeckAtom>(Vec1U{natlig}))
        , protein_h(onHost::allocHost<DeckAtom>(Vec1U{natpro}))
        , ff_h(onHost::allocHost<FFParams>(Vec1U{ffCount}))
        , rx_h(onHost::allocHost<float>(Vec1U{nposes}))
        , ry_h(onHost::allocHost<float>(Vec1U{nposes}))
        , rz_h(onHost::allocHost<float>(Vec1U{nposes}))
        , tx_h(onHost::allocHost<float>(Vec1U{nposes}))
        , ty_h(onHost::allocHost<float>(Vec1U{nposes}))
        , tz_h(onHost::allocHost<float>(Vec1U{nposes}))
        , out_h(onHost::allocHost<float>(Vec1U{nposes}))
        , ligand_d(onHost::allocLike(device, ligand_h))
        , protein_d(onHost::allocLike(device, protein_h))
        , ff_d(onHost::allocLike(device, ff_h))
        , rx_d(onHost::allocLike(device, rx_h))
        , ry_d(onHost::allocLike(device, ry_h))
        , rz_d(onHost::allocLike(device, rz_h))
        , tx_d(onHost::allocLike(device, tx_h))
        , ty_d(onHost::allocLike(device, ty_h))
        , tz_d(onHost::allocLike(device, tz_h))
        , out_d(onHost::allocLike(device, out_h))
        , threadSpec(makeThreadSpec())
    {
    }

    double init(
        std::vector<DeckAtom> const& ligand,
        std::vector<DeckAtom> const& protein,
        std::vector<FFParams> const& ff,
        std::array<std::vector<float>, 6> const& dof)
    {
        for(std::uint32_t i = 0; i < this->natlig; ++i)
            this->ligand_h[i] = ligand[i];
        for(std::uint32_t i = 0; i < this->natpro; ++i)
            this->protein_h[i] = protein[i];
        for(std::uint32_t i = 0; i < this->ffCount; ++i)
            this->ff_h[i] = ff[i];
        for(std::uint32_t i = 0; i < this->nposes; ++i)
        {
            this->rx_h[i] = dof[0][i];
            this->ry_h[i] = dof[1][i];
            this->rz_h[i] = dof[2][i];
            this->tx_h[i] = dof[3][i];
            this->ty_h[i] = dof[4][i];
            this->tz_h[i] = dof[5][i];
            this->out_h[i] = 0.0f;
        }

        auto const start = Clock::now();
        onHost::memcpy(this->queue, this->ligand_d, this->ligand_h);
        onHost::memcpy(this->queue, this->protein_d, this->protein_h);
        onHost::memcpy(this->queue, this->ff_d, this->ff_h);
        onHost::memcpy(this->queue, this->rx_d, this->rx_h);
        onHost::memcpy(this->queue, this->ry_d, this->ry_h);
        onHost::memcpy(this->queue, this->rz_d, this->rz_h);
        onHost::memcpy(this->queue, this->tx_d, this->tx_h);
        onHost::memcpy(this->queue, this->ty_d, this->ty_h);
        onHost::memcpy(this->queue, this->tz_d, this->tz_h);
        onHost::memset(this->queue, this->out_d, 0x00);
        onHost::wait(this->queue);
        auto const end = Clock::now();

        std::chrono::duration<double, std::milli> const dt = end - start;
        return dt.count();
    }

    RunTimings run_once()
    {
        onHost::wait(this->queue);

        auto const kernelStart = Clock::now();
        this->queue.enqueue(
            this->exec,
            this->threadSpec,
            ScorePosesKernel1D{},
            this->ligand_d,
            this->protein_d,
            this->ff_d,
            this->rx_d,
            this->ry_d,
            this->rz_d,
            this->tx_d,
            this->ty_d,
            this->tz_d,
            this->out_d,
            this->natlig,
            this->natpro,
            this->nposes);
        onHost::wait(this->queue);
        auto const kernelEnd = Clock::now();

        auto const d2hStart = Clock::now();
        onHost::memcpy(this->queue, this->out_h, this->out_d);
        onHost::wait(this->queue);
        auto const d2hEnd = Clock::now();

        return RunTimings{
            std::chrono::duration<double, std::milli>(kernelEnd - kernelStart).count(),
            std::chrono::duration<double, std::milli>(d2hEnd - d2hStart).count()};
    }

    void copy_energies(std::vector<float>& dest) const
    {
        dest.resize(this->nposes);
        for(std::uint32_t i = 0; i < this->nposes; ++i)
            dest[i] = this->out_h[i];
    }

    float accumulate_output() const
    {
        float sum = 0.0f;
        for(std::uint32_t i = 0; i < this->nposes; ++i)
            sum += this->out_h[i];
        return sum;
    }

private:
    static DeviceHandle makeDevice(Backend const& backend)
    {
        auto selector = onHost::makeDeviceSelector(backend[alpaka::object::deviceSpec]);
        return selector.makeDevice(0);
    }

    ThreadSpecType makeThreadSpec() const
    {
        std::uint32_t const threadsPerBlock = 256u;
        onHost::ThreadSpec spec{(nposes + threadsPerBlock - 1u) / threadsPerBlock, threadsPerBlock};
        if constexpr(alpaka::isSeqExecutor(std::decay_t<decltype(exec)>{}))
        {
            spec.m_numThreads = decltype(spec.m_numThreads){1u};
            spec.m_numBlocks = decltype(spec.m_numBlocks){nposes};
        }
        return spec;
    }
};
