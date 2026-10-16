/* Copyright 2026 Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file
 *
 * A kernel states the schedule of its blocks on the OpenMP executor (alpaka/core/OmpSchedule.hpp). Under either
 * schedule every block runs once, on every executor; the executors that are not OpenMP ignore the schedule.
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

/// The first thread of each block adds one to the counter of the block.
template<OmpSchedule T_Schedule>
struct CountBlocks
{
    static constexpr auto ompSchedule = T_Schedule;

    ALPAKA_FN_ACC void operator()(auto const& acc, auto counters) const
    {
        for(auto block : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{counters.getExtents()}))
        {
            if(onAcc::worker::threadsInBlock.getThreadIdx(acc).x() == 0u)
                onAcc::atomicAdd(acc, &counters[block], 1u);
        }
    }
};

/// A kernel that states nothing keeps the static schedule.
struct StatesNothing
{
    ALPAKA_FN_ACC void operator()(auto const&) const
    {
    }
};

static_assert(ompScheduleOf<StatesNothing> == OmpSchedule::Static);
static_assert(ompScheduleOf<CountBlocks<OmpSchedule::Dynamic>> == OmpSchedule::Dynamic);
static_assert(ompScheduleOf<CountBlocks<OmpSchedule::Static> const&> == OmpSchedule::Static);

template<OmpSchedule T_Schedule>
void requireEveryBlockOnce(auto deviceExec)
{
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    onHost::Queue queue = device.makeQueue();
    constexpr auto blocks = Vec{257u};
    auto counters = onHost::alloc<uint32_t>(device, blocks);
    onHost::memset(queue, counters, 0u);
    queue.enqueue(onHost::FrameSpec{blocks, Vec{1u}, exec}, KernelBundle{CountBlocks<T_Schedule>{}, counters});
    auto onHostCounters = onHost::allocHostLike(counters);
    onHost::memcpy(queue, onHostCounters, counters);
    onHost::wait(queue);
    meta::ndLoopIncIdx(blocks, [&](auto idx) { CHECK(onHostCounters[idx] == 1u); });
}

TEMPLATE_LIST_TEST_CASE("every block of a dynamic schedule runs once", "[kernel][ompSchedule]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    requireEveryBlockOnce<OmpSchedule::Dynamic>(deviceExec);
}

TEMPLATE_LIST_TEST_CASE("every block of a static schedule runs once", "[kernel][ompSchedule]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    requireEveryBlockOnce<OmpSchedule::Static>(deviceExec);
}

#if ALPAKA_OMP
#    include <omp.h>

#    include <chrono>

/** The first block waits until every other block has run, for two seconds at most, and every block writes the
 * OpenMP thread that runs it. Under a dynamic schedule the other threads of the team take every other block while
 * the first one waits, so the thread of the first block runs that block alone. Under a static schedule that thread
 * owns a contiguous part of the blocks and runs them after its wait.
 */
struct FirstBlockWaits
{
    static constexpr auto ompSchedule = OmpSchedule::Dynamic;

    ALPAKA_FN_ACC void operator()(auto const& acc, auto owners, auto done, uint32_t blocks) const
    {
        for(auto block : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{Vec{blocks}}))
        {
            owners[block] = static_cast<uint32_t>(::omp_get_thread_num());
            if(block.x() == 0u)
            {
                auto const start = std::chrono::steady_clock::now();
                while(onAcc::atomicAdd(acc, &done[Vec{0u}], 0u) < blocks - 1u
                      && std::chrono::steady_clock::now() - start < std::chrono::seconds{2})
                {
                }
            }
            else
                onAcc::atomicAdd(acc, &done[Vec{0u}], 1u);
        }
    }
};

TEST_CASE("the OpenMP executor shares the blocks of a dynamic kernel among the free threads", "[kernel][ompSchedule]")
{
    ::omp_set_dynamic(0);
    int const before = ::omp_get_max_threads();
    ::omp_set_num_threads(4);
    auto device = onHost::makeDeviceSelector(onHost::DeviceSpec{api::host, deviceKind::cpu}).makeDevice(0);
    auto queue = device.makeQueue(queueKind::blocking);
    constexpr uint32_t blocks = 256u;
    auto owners = onHost::alloc<uint32_t>(device, Vec{blocks});
    auto done = onHost::alloc<uint32_t>(device, Vec{1u});
    onHost::memset(queue, done, 0u);
    queue.enqueue(
        onHost::FrameSpec{Vec{blocks}, Vec{1u}, exec::cpuOmpBlocks},
        KernelBundle{FirstBlockWaits{}, owners, done, blocks});
    onHost::wait(queue);
    ::omp_set_num_threads(before);

    uint32_t const waiter = owners[Vec{0u}];
    uint32_t ranByWaiter = 0u;
    meta::ndLoopIncIdx(Vec{blocks}, [&](auto idx) { ranByWaiter += owners[idx] == waiter ? 1u : 0u; });
    CHECK(ranByWaiter == 1u);
}
#endif
