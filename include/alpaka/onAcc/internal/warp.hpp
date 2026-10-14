/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 *
 * Defines the Alpaka3 warp traits for kernel calls for votes and shuffles.
 * Central helpers (all, any, ballot, shfl…) forward to trait::Op specializations.
 *
 * Dispatch now depends on api::X and deviceKind::Y tags, keeping the layer backend free.
 * Provides a single entry point that works for host and GPU backends alike.
 *
 * Legacy Alpaka preferred the dispatch through ConceptWarp implementations per accelerator and used
 * interface::ImplementationBase indirection to select the correct implementation.
 */

#pragma once

#include "alpaka/api/api.hpp"
#include "alpaka/api/concepts/api.hpp"
#include "alpaka/api/cuda/computeApi.hpp"
#include "alpaka/api/hip/computeApi.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/api/unifiedCudaHip/ComputeApi.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/tag.hpp"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace alpaka::onAcc::warp::internal
{
    template<alpaka::onAcc::concepts::Acc T_Acc>
    constexpr uint32_t getSize()
    {
        return T_Acc::getWarpSize();
    }

    /** Retrieve a bit-mask describing which warp lanes are active. */
    struct Activemask
    {
        template<typename T_Acc, alpaka::concepts::Api T_Api>
        struct Op
        {
            constexpr auto operator()(T_Acc const&, T_Api) const
            {
                static_assert(
                    sizeof(T_Acc) && false,
                    "Missing warp Activemask implementation for the accelerator or the executor.");
                return 0u;
            }
        };
    };

    struct GetLaneIdx
    {
        template<typename T_Acc, alpaka::concepts::Api T_Api>
        struct Op
        {
            constexpr auto operator()(T_Acc const&, T_Api) const
            {
                static_assert(
                    sizeof(T_Acc) && false,
                    "Missing warp GetLaneIdx implementation for the accelerator or the executor.");
                return 0u;
            }
        };
    };

    /** The API a warp function runs on: the accelerator's, or the API of this compile pass for an executor.
     *
     * A warp function reads nothing of the accelerator on CUDA, HIP and the host; code without an accelerator
     * at hand, such as a function called from deep inside a kernel, passes the executor it was compiled for.
     * oneAPI reads the sub-group from the accelerator, and takes the accelerator alone.
     */
    constexpr auto apiOf(auto const& accOrExecutor)
    {
        if constexpr(alpaka::onAcc::concepts::Acc<ALPAKA_TYPEOF(accOrExecutor)>)
            return ALPAKA_TYPEOF(accOrExecutor[object::api]){};
        else
            return thisApi();
    }

    /// The device kind an API runs its warps on, for the APIs that have one kind of device.
    template<alpaka::concepts::Api T_Api>
    struct DeviceKindOf;

    template<>
    struct DeviceKindOf<api::Cuda>
    {
        using type = alpaka::deviceKind::NvidiaGpu;
    };

    template<>
    struct DeviceKindOf<api::Hip>
    {
        using type = alpaka::deviceKind::AmdGpu;
    };

    /** The size of a warp of a device kind, as the compute API of the kind states it.
     *
     * The trait is complete only in a compile pass of CUDA or HIP, where the unified compute API declares the size;
     * a host pass never asks for it, because the host API answers before this trait is read.
     */
    template<alpaka::concepts::DeviceKind T_DeviceKind>
    struct WarpSizeOfKind;

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
    template<alpaka::concepts::DeviceKind T_DeviceKind>
    requires requires { unifiedCudaHip::internal::WarpSize::Get<T_DeviceKind>{}(); }
    struct WarpSizeOfKind<T_DeviceKind>
    {
        static constexpr uint32_t value = unifiedCudaHip::internal::WarpSize::Get<T_DeviceKind>{}();
    };
#endif

    /// The size of a warp of an API, where the API alone decides it: one on the host, the device kind's on CUDA and
    /// HIP.
    template<alpaka::concepts::Api T_Api>
    constexpr uint32_t apiWarpSize()
    {
        if constexpr(std::is_same_v<T_Api, api::Host>)
            return 1u;
        else
            return WarpSizeOfKind<typename DeviceKindOf<T_Api>::type>::value;
    }

    /** The size of a warp: the accelerator's, or the API's of this compile pass for an executor.
     *
     * The API of the executor is read through the executor, so that the size is asked for where the function is
     * used, with every compute API declared, and not where this template is defined.
     */
    template<typename T_AccOrExecutor>
    constexpr uint32_t sizeOf()
    {
        if constexpr(alpaka::onAcc::concepts::Acc<T_AccOrExecutor>)
            return T_AccOrExecutor::getWarpSize();
        else
            return apiWarpSize<ALPAKA_TYPEOF(apiOf(std::declval<T_AccOrExecutor const&>()))>();
    }

    /** Return the lane index of the current thread within its warp. */
    constexpr uint32_t getLaneIdx(auto const& accOrExecutor)
    {
        using Acc = ALPAKA_TYPEOF(accOrExecutor);
        using Api = ALPAKA_TYPEOF(apiOf(accOrExecutor));
        return GetLaneIdx::Op<Acc, Api>{}(accOrExecutor, Api{});
    }

    struct GetWarpIdx
    {
        template<alpaka::onAcc::concepts::Acc T_Acc, alpaka::concepts::Api T_Api>
        struct Op
        {
            constexpr auto operator()(T_Acc const&, T_Api) const
            {
                static_assert(sizeof(T_Acc) && false, "Missing warp GetWarpIdx implementation for the accelerator.");
                return 0u;
            }
        };
    };

    /** Return the warp index within the block. */
    constexpr uint32_t getWarpIdx(alpaka::onAcc::concepts::Acc auto const& acc)
    {
        using Acc = ALPAKA_TYPEOF(acc);
        using Api = ALPAKA_TYPEOF(acc[object::api]);
        return GetWarpIdx::Op<Acc, Api>{}(acc, Api{});
    }

    struct All
    {
        template<typename T_Acc, alpaka::concepts::Api T_Api>
        struct Op
        {
            constexpr bool operator()(T_Acc const&, T_Api, int32_t predicate) const
            {
                alpaka::unused(predicate);
                static_assert(
                    sizeof(T_Acc) && false,
                    "Missing warp All implementation for the accelerator or the executor.");
                return false;
            }
        };
    };

    struct Any
    {
        template<typename T_Acc, alpaka::concepts::Api T_Api>
        struct Op
        {
            constexpr bool operator()(T_Acc const&, T_Api, int32_t predicate) const
            {
                alpaka::unused(predicate);
                static_assert(
                    sizeof(T_Acc) && false,
                    "Missing warp Any implementation for the accelerator or the executor.");
                return false;
            }
        };
    };

    struct Ballot
    {
        template<typename T_Acc, alpaka::concepts::Api T_Api>
        struct Op
        {
            constexpr auto operator()(T_Acc const&, T_Api, int32_t predicate) const
            {
                alpaka::unused(predicate);
                static_assert(
                    sizeof(T_Acc) && false,
                    "Missing warp Ballot implementation for the accelerator or the executor.");
                return 0;
            }
        };
    };

    struct Shfl
    {
        template<typename T_Acc, alpaka::concepts::Api T_Api, typename T>
        struct Op
        {
            constexpr T operator()(T_Acc const&, T_Api, T const& value, uint32_t srcLane, uint32_t width) const
            {
                alpaka::unused(value, srcLane, width);
                static_assert(
                    sizeof(T_Acc) && false,
                    "Missing warp Shfl implementation for the accelerator or the executor.");
                return T{};
            }
        };
    };

    struct ShflDown
    {
        template<typename T_Acc, alpaka::concepts::Api T_Api, typename T>
        struct Op
        {
            constexpr T operator()(T_Acc const&, T_Api, T const& value, uint32_t delta, uint32_t width) const
            {
                alpaka::unused(value, delta, width);
                static_assert(
                    sizeof(T_Acc) && false,
                    "Missing warp ShflDown implementation for the accelerator or the executor.");
                return T{};
            }
        };
    };

    struct ShflUp
    {
        template<typename T_Acc, alpaka::concepts::Api T_Api, typename T>
        struct Op
        {
            constexpr T operator()(T_Acc const&, T_Api, T const& value, uint32_t delta, uint32_t width) const
            {
                alpaka::unused(value, delta, width);
                static_assert(
                    sizeof(T_Acc) && false,
                    "Missing warp ShflUp implementation for the accelerator or the executor.");
                return T{};
            }
        };
    };

    struct ShflXor
    {
        template<typename T_Acc, alpaka::concepts::Api T_Api, typename T>
        struct Op
        {
            constexpr T operator()(T_Acc const&, T_Api, T const& value, uint32_t laneMask, uint32_t width) const
            {
                alpaka::unused(value, laneMask, width);
                static_assert(
                    sizeof(T_Acc) && false,
                    "Missing warp ShflXor implementation for the accelerator or the executor.");
                return T{};
            }
        };
    };
} // namespace alpaka::onAcc::warp::internal
