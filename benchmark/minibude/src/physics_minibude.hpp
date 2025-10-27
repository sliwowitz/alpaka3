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
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace onHost = alpaka::onHost;
using alpaka::Vec;

inline constexpr double kFlopsPerInteraction = 40.0; // Mirrors upstream miniBUDE
inline constexpr double kInstsPerInteraction = 25.0; // Mirrors upstream miniBUDE

// Optional tuning guards (default OFF). Enable via benchmark/minibude CMake options:
//  * MINIBUDE_USE_ALPAKA_SINCOS  (option MINIBUDE_USE_SINCOS)    -> combined sin/cos evaluations
//  * MINIBUDE_USE_FMA            (option MINIBUDE_USE_FMA)       -> fused multiply-add in hot paths
//  * MINIBUDE_ASSUME_ALIGNED     (option MINIBUDE_ASSUME_ALIGNED)-> align the small transform struct
//  * MINIBUDE_UNROLL_LIGAND      (option MINIBUDE_UNROLL_LIGAND) -> extra ligand loop unroll hint
// These toggles do not change default behaviour and can be combined for local benchmarking.

#if defined(MINIBUDE_ASSUME_ALIGNED)
struct alignas(64) MiniBudeTransform
#else
struct MiniBudeTransform
#endif
{
    float t00, t01, t02, t03;
    float t10, t11, t12, t13;
    float t20, t21, t22, t23;
};

// ===== Alpaka kernel and host wrapper to compute energies for all poses =====

template<std::uint32_t T_PPWI>
struct ScorePosesKernel1D
{
    // The `MdSpan` templates can be replaced with `MdSpan auto` types in nvcc 13.2+
    template<
        alpaka::concepts::MdSpan T_ligand,
        alpaka::concepts::MdSpan T_protein,
        alpaka::concepts::MdSpan T_ff,
        alpaka::concepts::MdSpan T_rx,
        alpaka::concepts::MdSpan T_ry,
        alpaka::concepts::MdSpan T_rz,
        alpaka::concepts::MdSpan T_tx,
        alpaka::concepts::MdSpan T_ty,
        alpaka::concepts::MdSpan T_tz,
        alpaka::concepts::MdSpan T_outE
    >
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        T_ligand const ligand,
        T_protein const protein,
        T_ff const ff,
        T_rx const rx,
        T_ry const ry,
        T_rz const rz,
        T_tx const tx,
        T_ty const ty,
        T_tz const tz,
        T_outE outE,
        std::uint32_t natlig,
        std::uint32_t natpro,
        std::uint32_t nposes) const
    {
        auto const threadIdxInGrid = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);
        auto const numThreadsInGrid = acc.getExtentsOf(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);

        auto const linearGridThreadIndex
            = static_cast<std::uint32_t>(alpaka::linearize(numThreadsInGrid, threadIdxInGrid));
        auto const linearGridSize = static_cast<std::uint32_t>(numThreadsInGrid.product());
        std::uint64_t const workStride = static_cast<std::uint64_t>(linearGridSize) * static_cast<std::uint64_t>(T_PPWI);

        for(std::uint64_t base = static_cast<std::uint64_t>(linearGridThreadIndex) * T_PPWI;
            base < nposes;
            base += workStride)
        {
#pragma unroll
            for(std::uint32_t j = 0u; j < T_PPWI; ++j)
            {
                std::uint32_t const i = static_cast<std::uint32_t>(base + j);
                if(i >= nposes)
                    break;

                float sx;
                float cx;
#if defined(MINIBUDE_USE_ALPAKA_SINCOS)
                alpaka::math::sincos(rx[i], sx, cx);
#else
                sx = alpaka::math::sin(rx[i]);
                cx = alpaka::math::cos(rx[i]);
#endif
                float sy;
                float cy;
#if defined(MINIBUDE_USE_ALPAKA_SINCOS)
                alpaka::math::sincos(ry[i], sy, cy);
#else
                sy = alpaka::math::sin(ry[i]);
                cy = alpaka::math::cos(ry[i]);
#endif
                float sz;
                float cz;
#if defined(MINIBUDE_USE_ALPAKA_SINCOS)
                alpaka::math::sincos(rz[i], sz, cz);
#else
                sz = alpaka::math::sin(rz[i]);
                cz = alpaka::math::cos(rz[i]);
#endif

                float t00 = cy * cz;
                float t01 = sx * sy * cz - cx * sz;
                float t02 = cx * sy * cz + sx * sz;
                float t03 = tx[i];
                float t10 = cy * sz;
                float t11 = sx * sy * sz + cx * cz;
                float t12 = cx * sy * sz - sx * cz;
                float t13 = ty[i];
                float t20 = -sy;
                float t21 = sx * cy;
                float t22 = cx * cy;
                float t23 = tz[i];

#if defined(MINIBUDE_ASSUME_ALIGNED)
                MiniBudeTransform transform{t00, t01, t02, t03, t10, t11, t12, t13, t20, t21, t22, t23};
#    define T00 transform.t00
#    define T01 transform.t01
#    define T02 transform.t02
#    define T03 transform.t03
#    define T10 transform.t10
#    define T11 transform.t11
#    define T12 transform.t12
#    define T13 transform.t13
#    define T20 transform.t20
#    define T21 transform.t21
#    define T22 transform.t22
#    define T23 transform.t23
#else
#    define T00 t00
#    define T01 t01
#    define T02 t02
#    define T03 t03
#    define T10 t10
#    define T11 t11
#    define T12 t12
#    define T13 t13
#    define T20 t20
#    define T21 t21
#    define T22 t22
#    define T23 t23
#endif

                float etot = 0.0f;

                for(std::uint32_t il = 0; il < natlig; ++il)
                {
                    DeckAtom const l_atom = ligand[il];
                    FFParams const l_params = ff[static_cast<std::uint32_t>(l_atom.type)];
                    bool const lhphb_ltz = (l_params.hphb < 0.f);
                    bool const lhphb_gtz = (l_params.hphb > 0.f);

#if defined(MINIBUDE_USE_FMA)
                    float const lpos_x = alpaka::math::fma(
                        l_atom.z,
                        T02,
                        alpaka::math::fma(l_atom.y, T01, alpaka::math::fma(l_atom.x, T00, T03)));
                    float const lpos_y = alpaka::math::fma(
                        l_atom.z,
                        T12,
                        alpaka::math::fma(l_atom.y, T11, alpaka::math::fma(l_atom.x, T10, T13)));
                    float const lpos_z = alpaka::math::fma(
                        l_atom.z,
                        T22,
                        alpaka::math::fma(l_atom.y, T21, alpaka::math::fma(l_atom.x, T20, T23)));
#else
                    float const lpos_x = T03 + l_atom.x * T00 + l_atom.y * T01 + l_atom.z * T02;
                    float const lpos_y = T13 + l_atom.x * T10 + l_atom.y * T11 + l_atom.z * T12;
                    float const lpos_z = T23 + l_atom.x * T20 + l_atom.y * T21 + l_atom.z * T22;
#endif

#if defined(MINIBUDE_UNROLL_LIGAND)
#    pragma unroll
#endif
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
                        int const zone1 = (distbb < ZERO);

                        float const steric_factor = zone1 ? (TWO * HARDNESS) : 0.f;
#if defined(MINIBUDE_USE_FMA)
                        etot = alpaka::math::fma((ONE - (distij * r_radij)), steric_factor, etot);
#else
                        etot += (ONE - (distij * r_radij)) * steric_factor;
#endif

                        float chrg_e = chrg_init * ((zone1 ? ONE : (ONE - distbb * elcdst1)) * ((distbb < elcdst) ? ONE : ZERO));
                        if(type_E)
                            chrg_e = -alpaka::math::abs(chrg_e);
#if defined(MINIBUDE_USE_FMA)
                        etot = alpaka::math::fma(chrg_e, CNSTNT, etot);
#else
                        etot += chrg_e * CNSTNT;
#endif

                        float const coeff = (ONE - (distbb * r_distdslv));
                        float dslv_e = dslv_init * (((distbb < distdslv) && phphb_nz) ? ONE : 0.f);
                        dslv_e *= (zone1 ? ONE : coeff);
#if defined(MINIBUDE_USE_FMA)
                        etot = alpaka::math::fma(dslv_e, 1.0f, etot);
#else
                        etot += dslv_e;
#endif
                    }
                }

                outE[i] = etot * HALF;
            }
        }
#if defined(MINIBUDE_ASSUME_ALIGNED)
#    undef T00
#    undef T01
#    undef T02
#    undef T03
#    undef T10
#    undef T11
#    undef T12
#    undef T13
#    undef T20
#    undef T21
#    undef T22
#    undef T23
#else
#    undef T00
#    undef T01
#    undef T02
#    undef T03
#    undef T10
#    undef T11
#    undef T12
#    undef T13
#    undef T20
#    undef T21
#    undef T22
#    undef T23
#endif
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

    struct ThreadSpecInfo
    {
        ThreadSpecType spec;
        std::uint32_t actualWgsize;
    };

    template<std::uint32_t T_PPWI>
    ThreadSpecInfo makeThreadSpec(std::uint32_t requestedWgsize) const
    {
        if constexpr(alpaka::exec::isSeqExecutor_v<ExecType>)
        {
            return ThreadSpecInfo{ThreadSpecType{nposes, 1u}, 1u};
        }
        else
        {
            std::uint32_t const actual = std::max(1u, requestedWgsize);
            std::uint64_t const workPerBlock = static_cast<std::uint64_t>(actual) * static_cast<std::uint64_t>(T_PPWI);
            std::uint32_t numBlocks = static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(nposes) + workPerBlock - 1u) / workPerBlock);
            if(numBlocks == 0u)
                numBlocks = 1u;
            return ThreadSpecInfo{ThreadSpecType{numBlocks, actual}, actual};
        }
    }

    template<std::uint32_t T_PPWI>
    RunTimings run(ThreadSpecType const& threadSpec)
    {
        onHost::wait(this->queue);

        auto const kernelStart = Clock::now();
        this->queue.enqueue(
            this->exec,
            threadSpec,
            ScorePosesKernel1D<T_PPWI>{},
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
};

inline constexpr std::array<std::uint32_t, 8> kSupportedPpwiValues{1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u};

inline bool isSupportedPpwi(std::uint32_t value)
{
    return std::find(kSupportedPpwiValues.begin(), kSupportedPpwiValues.end(), value) != kSupportedPpwiValues.end();
}

template<typename Fn>
auto dispatchPpwi(std::uint32_t value, Fn&& fn)
{
    switch(value)
    {
    case 1u:
        return fn(std::integral_constant<std::uint32_t, 1u>{});
    case 2u:
        return fn(std::integral_constant<std::uint32_t, 2u>{});
    case 4u:
        return fn(std::integral_constant<std::uint32_t, 4u>{});
    case 8u:
        return fn(std::integral_constant<std::uint32_t, 8u>{});
    case 16u:
        return fn(std::integral_constant<std::uint32_t, 16u>{});
    case 32u:
        return fn(std::integral_constant<std::uint32_t, 32u>{});
    case 64u:
        return fn(std::integral_constant<std::uint32_t, 64u>{});
    case 128u:
        return fn(std::integral_constant<std::uint32_t, 128u>{});
    default:
        throw std::runtime_error("Unsupported PPWI value");
    }
}
