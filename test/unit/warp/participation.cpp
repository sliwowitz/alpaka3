/* Copyright 2026 Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

#include "utils.hpp"

#include <alpaka/onAcc/warp.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <type_traits>

using namespace alpaka;
using alpaka::test::warp::warpCheck;
using alpaka::test::warp::WarpTestBackends;

namespace
{
    /** Every lane of every warp calls each function under the promise of the whole warp, with the accelerator and
     * with the executor, and reads what the active lanes form gives. The block is a whole number of warps, so the
     * promise holds.
     */
    struct WholeWarpKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(TAcc const& acc, concepts::IMdSpan<bool> auto success) const
        {
            namespace warp = onAcc::warp;
            constexpr auto whole = warp::participation::wholeWarp;
            auto const lane = static_cast<std::int32_t>(warp::getLaneIdx(acc));
            warpCheck(success, warp::all(acc, whole, 1) == warp::all(acc, 1));
            warpCheck(success, warp::any(acc, whole, lane == 0) == warp::any(acc, lane == 0));
            warpCheck(success, warp::ballot(acc, whole, lane % 2) == warp::ballot(acc, lane % 2));
            warpCheck(success, warp::shfl(acc, whole, lane, 0u) == warp::shfl(acc, lane, 0u));
            warpCheck(success, warp::shflXor(acc, whole, lane, 1u) == warp::shflXor(acc, lane, 1u));
            warpCheck(success, warp::shflDown(acc, whole, lane, 1u) == warp::shflDown(acc, lane, 1u));
            warpCheck(success, warp::shflUp(acc, whole, lane, 1u) == warp::shflUp(acc, lane, 1u));
            warpCheck(
                success,
                warp::shflXor(acc, whole, 4.0f - static_cast<float>(lane), 1u)
                    == warp::shflXor(acc, 4.0f - static_cast<float>(lane), 1u));
            using Api = ALPAKA_TYPEOF(acc[object::api]);
            if constexpr(!std::is_same_v<Api, api::OneApi>)
            {
                auto const exec = acc[object::exec];
                warpCheck(success, warp::ballot(exec, whole, lane % 2) == warp::ballot(acc, lane % 2));
                warpCheck(success, warp::shfl(exec, whole, lane, 0u) == warp::shfl(acc, lane, 0u));
                warpCheck(success, warp::shflXor(exec, whole, lane, 1u) == warp::shflXor(acc, lane, 1u));
            }
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE(
    "warp functions under the promise of the whole warp",
    "[warp][participation]",
    WarpTestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    auto deviceProperties = device.getDeviceProperties();
    auto const warpExtent = deviceProperties.warpSize;
    auto queue = device.makeQueue(queueKind::blocking);

    auto successHost = onHost::allocHost<bool>(1u);
    auto successDev = onHost::allocLike(device, successHost);
    auto const blocks = Vec<std::uint32_t, 1u>{2u};
    auto const threads = Vec<std::uint32_t, 1u>{2u * warpExtent};
    onHost::memset(queue, successDev, static_cast<std::uint8_t>(true));
    queue.enqueue(onHost::FrameSpec{blocks, threads, exec}, KernelBundle{WholeWarpKernel{}, successDev});
    onHost::memcpy(queue, successHost, successDev);
    onHost::wait(queue);
    CHECK(successHost[0]);
}
