/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cuda/Api.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/internal/warp.hpp"

#include <cstdint>
#include <type_traits>

#if ALPAKA_LANG_CUDA
namespace alpaka::onAcc::warp::internal
{
    /// The mask of the lanes that take part: the active lanes as the device names them, or the whole warp.
    inline __device__ unsigned maskOf(participation::ActiveLanes)
    {
        return __activemask();
    }

    constexpr __device__ unsigned maskOf(participation::WholeWarp)
    {
        return 0xffff'ffffu;
    }

    template<typename T_Acc>
    struct Activemask::Op<T_Acc, api::Cuda>
    {
        constexpr __device__ auto operator()(T_Acc const&, api::Cuda) const
        {
            return __activemask();
        }
    };

    template<typename T_Acc>
    struct GetLaneIdx::Op<T_Acc, api::Cuda>
    {
        constexpr __device__ auto operator()(T_Acc const&, api::Cuda) const
        {
            unsigned lIdx;
#    if ALPAKA_COMP_NVCC
            asm volatile("mov.u32 %0, %%laneid;" : "=r"(lIdx));
#    else
            asm("mov.u32 %0, %%laneid;" : "=r"(lIdx));
#    endif
            return lIdx;
        }
    };

    template<alpaka::onAcc::concepts::Acc T_Acc>
    struct GetWarpIdx::Op<T_Acc, api::Cuda>
    {
        constexpr __device__ uint32_t operator()(T_Acc const& acc, api::Cuda) const
        {
            constexpr uint32_t warpExtent = onAcc::warp::internal::getSize<ALPAKA_TYPEOF(acc)>();
            alpaka::concepts::Vector auto blockThreadCount
                = acc.getExtentsOf(onAcc::origin::block, onAcc::unit::threads);
            alpaka::concepts::Vector auto threadIdxInBlock
                = acc.getIdxWithin(alpaka::onAcc::origin::block, alpaka::onAcc::unit::threads);
            return linearize(blockThreadCount, threadIdxInBlock) / warpExtent;
        }
    };

    template<typename T_Acc>
    struct All::Op<T_Acc, api::Cuda>
    {
        constexpr __device__ bool operator()(
            T_Acc const&,
            api::Cuda,
            Participation auto participation,
            int32_t predicate) const
        {
            return __all_sync(maskOf(participation), static_cast<int>(predicate)) != 0;
        }
    };

    template<typename T_Acc>
    struct Any::Op<T_Acc, api::Cuda>
    {
        constexpr __device__ bool operator()(
            T_Acc const&,
            api::Cuda,
            Participation auto participation,
            int32_t predicate) const
        {
            return __any_sync(maskOf(participation), static_cast<int>(predicate)) != 0;
        }
    };

    template<typename T_Acc>
    struct Ballot::Op<T_Acc, api::Cuda>
    {
        constexpr __device__ auto operator()(
            T_Acc const&,
            api::Cuda,
            Participation auto participation,
            int32_t predicate) const
        {
            return __ballot_sync(maskOf(participation), static_cast<int>(predicate));
        }
    };

    template<typename T_Acc, typename T>
    struct Shfl::Op<T_Acc, api::Cuda, T>
    {
        constexpr __device__ T operator()(
            T_Acc const&,
            api::Cuda,
            Participation auto participation,
            T const& value,
            uint32_t srcLane,
            uint32_t width) const
        {
            return __shfl_sync(maskOf(participation), value, static_cast<int>(srcLane), static_cast<int>(width));
        }
    };

    template<typename T_Acc, typename T>
    struct ShflDown::Op<T_Acc, api::Cuda, T>
    {
        constexpr __device__ T operator()(
            T_Acc const&,
            api::Cuda,
            Participation auto participation,
            T const& value,
            uint32_t delta,
            uint32_t width) const
        {
            return __shfl_down_sync(maskOf(participation), value, static_cast<int>(delta), static_cast<int>(width));
        }
    };

    template<typename T_Acc, typename T>
    struct ShflUp::Op<T_Acc, api::Cuda, T>
    {
        constexpr __device__ T operator()(
            T_Acc const&,
            api::Cuda,
            Participation auto participation,
            T const& value,
            uint32_t delta,
            uint32_t width) const
        {
            return __shfl_up_sync(maskOf(participation), value, static_cast<int>(delta), static_cast<int>(width));
        }
    };

    template<typename T_Acc, typename T>
    struct ShflXor::Op<T_Acc, api::Cuda, T>
    {
        constexpr __device__ T operator()(
            T_Acc const&,
            api::Cuda,
            Participation auto participation,
            T const& value,
            uint32_t laneMask,
            uint32_t width) const
        {
            return __shfl_xor_sync(maskOf(participation), value, static_cast<int>(laneMask), static_cast<int>(width));
        }
    };
} // namespace alpaka::onAcc::warp::internal
#endif
