/* Copyright 2026 Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file
 *
 * A device reports how many threads of an executor run at the same time: one thread for the serial executor, the
 * team of the calling thread for the OpenMP executor, and at least one thread for every executor.
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <concepts>
#include <type_traits>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("a device reports how many threads of an executor run at once", "[device][concurrency]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto executor = test::getExecutor(deviceExec);

    CHECK(device.getConcurrency(executor) >= 1u);
    if constexpr(std::same_as<ALPAKA_TYPEOF(executor), exec::CpuSerial>)
        CHECK(device.getConcurrency(executor) == 1u);
#if ALPAKA_OMP
    if constexpr(std::same_as<ALPAKA_TYPEOF(executor), exec::CpuOmpBlocks>)
    {
        int const before = ::omp_get_max_threads();
        ::omp_set_num_threads(3);
        CHECK(device.getConcurrency(executor) == 3u);
        ::omp_set_num_threads(before);
    }
#endif
}
