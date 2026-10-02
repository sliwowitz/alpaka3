/* Copyright 2026 Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <type_traits>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

/** Every element of a block adds one into two counters in the memory the block shares, with the scope that says so.
 * The first element writes the counts of its block.
 */
struct SharedCountKernel
{
    template<typename T_Acc>
    ALPAKA_FN_ACC void operator()(T_Acc const& acc, auto out, auto numBlocks, auto blockExtents) const
    {
        auto& real = onAcc::declareSharedVar<float, uniqueId()>(acc);
        auto& whole = onAcc::declareSharedVar<std::uint32_t, uniqueId()>(acc);

        for(auto blockIdx : onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, IdxRange{numBlocks}))
        {
            for(auto inBlock : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{blockExtents}))
            {
                if(inBlock.x() == 0u)
                {
                    real = 0.0f;
                    whole = 0u;
                }
            }
            onAcc::syncBlockThreads(acc);
            for(auto inBlock : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{blockExtents}))
            {
                alpaka::unused(inBlock);
                onAcc::atomicAdd(acc, &real, 1.0f, onAcc::scope::blockShared);
                onAcc::atomicAdd(acc, &whole, 1u, onAcc::scope::blockShared);
            }
            onAcc::syncBlockThreads(acc);
            for(auto inBlock : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{blockExtents}))
            {
                if(inBlock.x() == 0u)
                {
                    out[Vec{2u * blockIdx.x()}] = static_cast<std::uint32_t>(real);
                    out[Vec{2u * blockIdx.x() + 1u}] = whole;
                }
            }
            onAcc::syncBlockThreads(acc);
        }
    }
};

TEMPLATE_LIST_TEST_CASE("atomic add into the memory a block shares", "[atomic][sharedMem]", TestApis)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());
    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);
    onHost::Queue queue = device.makeQueue();

    constexpr Vec numBlocks = Vec{4u};
    constexpr Vec blockExtent = Vec{128u};
    auto dBuff = onHost::alloc<std::uint32_t>(device, Vec{2u * numBlocks.x()});
    auto hBuff = onHost::allocHostLike(dBuff);

    queue.enqueue(
        onHost::FrameSpec{numBlocks, blockExtent, exec},
        KernelBundle{SharedCountKernel{}, dBuff, numBlocks, blockExtent});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);

    auto* counts = onHost::data(hBuff);
    for(std::uint32_t i = 0u; i < 2u * numBlocks.x(); ++i)
        CHECK(counts[i] == blockExtent.x());
}
