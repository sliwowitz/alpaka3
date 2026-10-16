/* Copyright 2026 Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file
 *
 * The traversal by runs of a one-dimensional index range (onAcc::traverse::runs). Every index of the range is in
 * exactly one run of one worker, and no index outside the range is. With the contiguous layout a worker has one run
 * at most; with the strided layout every run has one index.
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

/// Each worker counts its runs, and adds one to the counter of every index in its runs.
template<onAcc::concepts::IdxMapping T_IdxLayout>
struct CountRuns
{
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        auto counters,
        auto runsOfWorkers,
        auto lengthsOfWorkers,
        uint32_t begin,
        uint32_t end,
        uint32_t stride) const
    {
        uint32_t runs = 0u;
        uint32_t longest = 0u;
        for(auto run : onAcc::makeIdxMap(
                acc,
                onAcc::worker::linearThreadsInGrid,
                IdxRange{Vec{begin}, Vec{end}, Vec{stride}},
                onAcc::traverse::runs,
                T_IdxLayout{}))
        {
            ++runs;
            uint32_t length = 0u;
            for(uint32_t at = run.getBeginMd().x(); at < run.getEndMd().x(); at += run.getStrideMd().x())
            {
                onAcc::atomicAdd(acc, &counters[Vec{at}], 1u);
                ++length;
            }
            longest = length > longest ? length : longest;
        }
        auto const worker = onAcc::worker::linearThreadsInGrid.getThreadIdx(acc);
        runsOfWorkers[worker] = runs;
        lengthsOfWorkers[worker] = longest;
    }
};

template<onAcc::concepts::IdxMapping T_IdxLayout>
void requireRuns(auto deviceExec, uint32_t begin, uint32_t end, uint32_t stride)
{
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    onHost::Queue queue = device.makeQueue();

    constexpr uint32_t indices = 128u;
    constexpr auto frames = Vec{4u};
    constexpr auto frameExtent = Vec{8u};
    constexpr uint32_t workersAtMost = 32u;

    auto counters = onHost::alloc<uint32_t>(device, Vec{indices});
    auto runsOfWorkers = onHost::alloc<uint32_t>(device, Vec{workersAtMost});
    auto lengthsOfWorkers = onHost::alloc<uint32_t>(device, Vec{workersAtMost});
    onHost::memset(queue, counters, 0u);
    onHost::memset(queue, runsOfWorkers, 0u);
    onHost::memset(queue, lengthsOfWorkers, 0u);
    queue.enqueue(
        onHost::FrameSpec{frames, frameExtent, exec},
        KernelBundle{CountRuns<T_IdxLayout>{}, counters, runsOfWorkers, lengthsOfWorkers, begin, end, stride});

    auto hostCounters = onHost::allocHostLike(counters);
    auto hostRuns = onHost::allocHostLike(runsOfWorkers);
    auto hostLengths = onHost::allocHostLike(lengthsOfWorkers);
    onHost::memcpy(queue, hostCounters, counters);
    onHost::memcpy(queue, hostRuns, runsOfWorkers);
    onHost::memcpy(queue, hostLengths, lengthsOfWorkers);
    onHost::wait(queue);

    uint32_t inRange = 0u;
    meta::ndLoopIncIdx(
        Vec{indices},
        [&](auto idx)
        {
            uint32_t const at = idx.x();
            bool const member = at >= begin && at < end && (at - begin) % stride == 0u;
            inRange += member ? 1u : 0u;
            CHECK(hostCounters[idx] == (member ? 1u : 0u));
        });
    uint32_t totalRuns = 0u;
    meta::ndLoopIncIdx(
        Vec{workersAtMost},
        [&](auto idx)
        {
            totalRuns += hostRuns[idx];
            if constexpr(std::is_same_v<T_IdxLayout, onAcc::layout::Contiguous>)
                CHECK(hostRuns[idx] <= 1u);
            if constexpr(std::is_same_v<T_IdxLayout, onAcc::layout::Strided>)
                CHECK(hostLengths[idx] <= 1u);
        });
    if constexpr(std::is_same_v<T_IdxLayout, onAcc::layout::Strided>)
        CHECK(totalRuns == inRange);
}

TEMPLATE_LIST_TEST_CASE("every index of a range is in one run of one worker", "[kernel][traverse][runs]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    // A range with an offset, and a range with an offset and a stride.
    for(auto const [begin, end, stride] : {std::tuple{3u, 103u, 1u}, std::tuple{1u, 127u, 3u}, std::tuple{5u, 6u, 1u}})
    {
        requireRuns<onAcc::layout::Contiguous>(deviceExec, begin, end, stride);
        requireRuns<onAcc::layout::Strided>(deviceExec, begin, end, stride);
        requireRuns<onAcc::layout::Optimized>(deviceExec, begin, end, stride);
    }
}
