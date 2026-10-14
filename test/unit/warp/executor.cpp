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
    /** Every warp function called with the executor gives what it gives with the accelerator. The kernel reads
     * the executor from the accelerator, as code deep inside a kernel knows the executor it was compiled for.
     */
    struct ExecutorFormsKernel
    {
        template<typename TAcc>
        ALPAKA_FN_ACC void operator()(TAcc const& acc, concepts::IMdSpan<bool> auto success) const
        {
            auto const exec = acc[object::exec];
            using Api = ALPAKA_TYPEOF(acc[object::api]);
            if constexpr(std::is_same_v<Api, api::OneApi>)
            {
                // oneAPI reads the sub-group from the accelerator; the executor form is not offered there.
                return;
            }
            else
            {
                warpCheck(success, onAcc::warp::getSize(exec) == onAcc::warp::getSize(acc));
                warpCheck(success, onAcc::warp::getSize<ALPAKA_TYPEOF(exec)>() == onAcc::warp::getSize<TAcc>());
                warpCheck(success, onAcc::warp::activemask(exec) == onAcc::warp::activemask(acc));
                auto const lane = static_cast<std::int32_t>(onAcc::warp::getLaneIdx(acc));
                warpCheck(success, static_cast<std::int32_t>(onAcc::warp::getLaneIdx(exec)) == lane);
                warpCheck(success, onAcc::warp::all(exec, 1) == onAcc::warp::all(acc, 1));
                warpCheck(success, onAcc::warp::any(exec, lane == 0) == onAcc::warp::any(acc, lane == 0));
                warpCheck(success, onAcc::warp::ballot(exec, lane % 2) == onAcc::warp::ballot(acc, lane % 2));
                warpCheck(success, onAcc::warp::shfl(exec, lane, 0u) == onAcc::warp::shfl(acc, lane, 0u));
                warpCheck(success, onAcc::warp::shflXor(exec, lane, 1u) == onAcc::warp::shflXor(acc, lane, 1u));
                warpCheck(success, onAcc::warp::shflDown(exec, lane, 1u) == onAcc::warp::shflDown(acc, lane, 1u));
                warpCheck(success, onAcc::warp::shflUp(exec, lane, 1u) == onAcc::warp::shflUp(acc, lane, 1u));
                warpCheck(
                    success,
                    onAcc::warp::shflXor(exec, 4.0f - static_cast<float>(lane), 1u)
                        == onAcc::warp::shflXor(acc, 4.0f - static_cast<float>(lane), 1u));
            }
        }
    };
} // namespace

TEMPLATE_LIST_TEST_CASE(
    "warp functions take the executor in place of the accelerator",
    "[warp][executor]",
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
    queue.enqueue(onHost::FrameSpec{blocks, threads, exec}, KernelBundle{ExecutorFormsKernel{}, successDev});
    onHost::memcpy(queue, successHost, successDev);
    onHost::wait(queue);
    CHECK(successHost[0]);
}
