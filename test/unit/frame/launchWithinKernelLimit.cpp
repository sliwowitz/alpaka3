/* Copyright 2026 Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file
 *
 * This test validates that a kernel which needs many registers can be called with a FrameSpec.
 *
 * The threads of a GPU block share the registers of the block. A kernel that needs many registers can therefore not
 * run in a block with as many threads as the frame has elements, and the launch must take fewer threads.
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

/** A kernel that keeps many values at once.
 *
 * Each element mixes numLive values in rounds. A round reads every value twice, and the kernel does not know the
 * number of rounds when it is compiled. Each value therefore keeps a register of its own over the whole loop.
 * The mean of two neighbours on a ring keeps the sum of the values, so the result is the sum of the first values.
 */
struct ManyLiveValuesKernel
{
    static constexpr uint32_t numLive = 192u;

    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        alpaka::concepts::IMdSpan auto out,
        alpaka::concepts::Vector auto extents,
        uint32_t rounds) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange(extents)))
        {
            out[i] = mixed(firstValue(i.x()), rounds, std::make_index_sequence<numLive>{});
        }
    }

    ALPAKA_FN_HOST_ACC static constexpr float firstValue(uint32_t elementIdx)
    {
        return static_cast<float>(elementIdx % 7u);
    }

    /// The result of an element. The values are small dyadic numbers, so the sum is exact.
    static constexpr float expected(uint32_t elementIdx)
    {
        return static_cast<float>(numLive) * firstValue(elementIdx)
               + static_cast<float>(numLive * (numLive - 1u) / 2u);
    }

private:
    template<std::size_t... T_k>
    ALPAKA_FN_ACC static float mixed(float first, uint32_t rounds, std::index_sequence<T_k...>)
    {
        float live[numLive] = {(first + static_cast<float>(T_k))...};
        for(uint32_t round = 0u; round < rounds; ++round)
        {
            float const next[numLive] = {(0.5f * (live[T_k] + live[(T_k + 1u) % numLive]))...};
            ((live[T_k] = next[T_k]), ...);
        }
        return (live[T_k] + ...);
    }
};

TEMPLATE_LIST_TEST_CASE("frame launch within the thread limit of the kernel", "", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    onHost::Queue queue = device.makeQueue();

    // The frame extent that the algorithms of the library take on a GPU.
    Vec const frameExtents{512u};
    Vec const extents{8u * frameExtents.x()};
    constexpr uint32_t rounds = 4u;

    auto dBuff = onHost::alloc<float>(device, extents);
    auto hBuff = onHost::allocHostLike(dBuff);

    queue.enqueue(
        onHost::FrameSpec{divExZero(extents, frameExtents), frameExtents, exec},
        KernelBundle{ManyLiveValuesKernel{}, dBuff, extents, rounds});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);

    for(uint32_t i = 0u; i < extents.x(); ++i)
        CHECK(hBuff[Vec{i}] == ManyLiveValuesKernel::expected(i));
}
