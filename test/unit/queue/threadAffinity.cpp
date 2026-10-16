/* Copyright 2026 Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file
 *
 * A launch on a blocking queue of the host device of all CPU domains leaves the affinity of the thread that enqueues
 * it as it was. The OpenMP runtime binds the threads of a team (OMP_PROC_BIND, OMP_PLACES), and the cpuset of all
 * domains holds every core: a launch that binds its threads to that cpuset removes the binding.
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#if defined(__linux__)
#    include <sched.h>

using namespace alpaka;

namespace
{
    struct EmptyKernel
    {
        ALPAKA_FN_ACC void operator()(auto const&) const
        {
        }
    };

    cpu_set_t affinityOfThisThread()
    {
        cpu_set_t set;
        CPU_ZERO(&set);
        REQUIRE(sched_getaffinity(0, sizeof(set), &set) == 0);
        return set;
    }

    void launchOnBlockingQueue(alpaka::concepts::Executor auto executor)
    {
        auto device = onHost::makeDeviceSelector(onHost::DeviceSpec{api::host, deviceKind::cpu}).makeDevice(0);
        auto queue = device.makeQueue(queueKind::blocking);
        queue.enqueue(onHost::FrameSpec{Vec{4u}, Vec{1u}, executor}, KernelBundle{EmptyKernel{}});
    }

    void requireAffinityKept(alpaka::concepts::Executor auto executor)
    {
        cpu_set_t const before = affinityOfThisThread();
        int first = 0;
        while(first < CPU_SETSIZE && !CPU_ISSET(first, &before))
            ++first;
        REQUIRE(first < CPU_SETSIZE);
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(first, &one);
        REQUIRE(sched_setaffinity(0, sizeof(one), &one) == 0);

        launchOnBlockingQueue(executor);
        cpu_set_t const after = affinityOfThisThread();
        REQUIRE(sched_setaffinity(0, sizeof(before), &before) == 0);
        CHECK(CPU_EQUAL(&one, &after));
    }
} // namespace

TEST_CASE("a blocking launch on the host device of all domains keeps the affinity of the serial thread", "[queue][affinity]")
{
    requireAffinityKept(exec::cpuSerial);
}

#    if ALPAKA_OMP
TEST_CASE("a blocking launch on the host device of all domains keeps the affinity of the OpenMP team", "[queue][affinity]")
{
    requireAffinityKept(exec::cpuOmpBlocks);
}
#    endif
#endif
