/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/simd/simdConfig.hpp"
#include "alpaka/utility.hpp"

#include <cstdint>
#include <type_traits>

namespace alpaka::onHost::internal
{

    /** The width in bytes of the native SIMD register that std::simd knows, or 0 without std::simd.
     *
     * The width is a fact of the architecture and not of the element type, so one element type asks for it: a
     * 32-bit one, whose vector type every standard library forms in a moment. The vector type of a 64-bit element
     * is what clang 22 instantiates for minutes in the device pass of a CUDA translation unit, through libstdc++
     * 15's experimental simd, so no 64-bit element type reaches std::simd from here.
     */
    constexpr size_t stdSimdRegisterBytes()
    {
#if ALPAKA_HAS_STD_SIMD
        return alpakaStdSimd::native_simd<float>::size() * sizeof(float);
#else
        return 0;
#endif
    }

    /** SIMD width in bytes defined by std::simd
     *
     * @return 0 if std::simd is not supported or the T_Type is unsupported, else the SIMD width in bytes
     */
    template<typename T_Type>
    constexpr size_t stdSimdWidth()
    {
        if constexpr(std::is_arithmetic_v<T_Type> && !std::is_same_v<T_Type, bool>)
            return stdSimdRegisterBytes();
        else
            return 0;
    }

    template<typename T_Type>
    constexpr uint32_t getCPUSimdWidth()
    {
        constexpr size_t possibleSimdWidthBytes =
#if defined(__AVX512BW__) || defined(__AVX512F__) || defined(__AVX512DQ__) || defined(__AVX512VL__)
            64u;
#elif defined(__riscv_vector)
            64u;
#elif defined(__riscv)
            // do not use vectors if the vector extension is not set
            sizeof(T_Type);
#elif defined(__AVX2__)
            32u;
#elif defined(__SSE__) || defined(__SSE2__) || defined(__SSE4_1__) || defined(__SSE4_2__)
            16u;
// Macro to be define by the user to enable SVE backend and specify SVE size
#elif defined(SVE_VECTOR_BITS)
            SVE_VECTOR_BITS / 8;
// If user has specified SVE vector length using the flag -msve-vector-bits
#elif defined(__ARM_FEATURE_SVE_BITS)
            __ARM_FEATURE_SVE_BITS / 8;
// ARM e.g. nvidia grace hopper
#elif defined(__ARM_FEATURE_SVE2_AES)
            16u;
// ARM e.g AWS Graviton 3
#elif defined(__ARM_FEATURE_SVE)
            32u;
#elif defined(__ARM_NEON)
            16u;
#elif defined(__ALTIVEC__)
            16u;
#else
            sizeof(T_Type);
#endif

        // we assume that the standard is maintaining the vector length better than we, therefore take it if vector
        // types are supported
        constexpr size_t simdWidthInByte = stdSimdWidth<T_Type>() ? stdSimdWidth<T_Type>() : possibleSimdWidthBytes;

        return alpaka::divExZero(simdWidthInByte, sizeof(T_Type));
    }

    constexpr uint32_t getCPUNumPipelines()
    {
        /* INTEL can issue 4 commands and AMD typically 2, since we can not distinguish between both we use
         * the higher value.
         * ARM SVE can typically issue 4 commands too.
         *
         * Therefore we use at the moment as default 4.
         */
        constexpr uint32_t numPipes = 4u;
        return numPipes;
    }

    constexpr uint32_t getCPUCachelineSize()
    {
        constexpr uint32_t cachlineBytes =
#ifdef __cpp_lib_hardware_interference_size
            std::hardware_constructive_interference_size;

#else
            // Fallback value, typically 64 bytes
            64;
#endif
        return cachlineBytes;
    }

} // namespace alpaka::onHost::internal
