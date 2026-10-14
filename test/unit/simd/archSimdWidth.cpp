/* Copyright 2026 Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>

using namespace alpaka;

/* The SIMD width of the host is the width of its register, a fact of the architecture: every arithmetic element
 * type holds as many lanes as fit into that one register, so the width in bytes is the same for all of them, and
 * no element type asks the standard library for a vector type of its own. A bool is not a lane. */
TEMPLATE_TEST_CASE(
    "the SIMD width of the host is one register whatever the arithmetic element type",
    "[simd][host]",
    float,
    double,
    std::int8_t,
    std::uint8_t,
    std::int32_t,
    std::uint32_t,
    std::int64_t,
    std::uint64_t)
{
    constexpr std::uint32_t lanes = alpaka::getArchSimdWidth<TestType>(api::host, deviceKind::cpu);
    constexpr std::size_t bytes = static_cast<std::size_t>(lanes) * sizeof(TestType);
    constexpr std::size_t registerBytes
        = static_cast<std::size_t>(alpaka::getArchSimdWidth<float>(api::host, deviceKind::cpu)) * sizeof(float);
    STATIC_REQUIRE(lanes >= 1u);
    STATIC_REQUIRE(bytes == registerBytes);
}

TEST_CASE("a bool has one lane of its own size on the host", "[simd][host]")
{
    STATIC_REQUIRE(alpaka::getArchSimdWidth<bool>(api::host, deviceKind::cpu) >= 1u);
}
