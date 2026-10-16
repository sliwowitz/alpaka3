/* Copyright 2026 Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file
 *
 * The OpenMP executor opens a team of no more threads than a launch has blocks, and runs a launch of one block on
 * the calling thread, outside any parallel region.
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#if ALPAKA_OMP
#    include <omp.h>

using namespace alpaka;

namespace
{
    /// Each block writes the size of the team that runs it, and whether a parallel region runs it.
    struct RecordTeam
    {
        ALPAKA_FN_ACC void operator()(auto const& acc, auto teamSizes, auto inRegion) const
        {
            for(auto block : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{teamSizes.getExtents()}))
            {
                teamSizes[block] = static_cast<uint32_t>(::omp_get_num_threads());
                inRegion[block] = static_cast<uint32_t>(::omp_in_parallel());
            }
        }
    };

    void requireTeam(uint32_t blocks, uint32_t expectedTeam, uint32_t expectedInRegion)
    {
        auto device = onHost::makeDeviceSelector(onHost::DeviceSpec{api::host, deviceKind::cpu}).makeDevice(0);
        auto queue = device.makeQueue(queueKind::blocking);
        auto teamSizes = onHost::alloc<uint32_t>(device, Vec{blocks});
        auto inRegion = onHost::alloc<uint32_t>(device, Vec{blocks});
        queue.enqueue(
            onHost::FrameSpec{Vec{blocks}, Vec{1u}, exec::cpuOmpBlocks},
            KernelBundle{RecordTeam{}, teamSizes, inRegion});
        onHost::wait(queue);
        meta::ndLoopIncIdx(
            Vec{blocks},
            [&](auto idx)
            {
                CHECK(teamSizes[idx] == expectedTeam);
                CHECK(inRegion[idx] == expectedInRegion);
            });
    }
} // namespace

TEST_CASE("the OpenMP executor opens a team of no more threads than a launch has blocks", "[queue][omp]")
{
    ::omp_set_dynamic(0);
    ::omp_set_num_threads(4);
    // One block runs on the calling thread, with no region.
    requireTeam(1u, 1u, 0u);
    // Two blocks wake two threads, not four.
    requireTeam(2u, 2u, 1u);
    // More blocks than threads take the whole team.
    requireTeam(64u, 4u, 1u);
}
#endif
