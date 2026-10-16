/* Copyright 2026 Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <type_traits>

namespace alpaka
{
    /** How the OpenMP executor shares the blocks of a launch among the threads of its team.
     *
     * A kernel states its schedule with a static data member `ompSchedule`. Without one, the threads share the
     * blocks statically, in equal contiguous parts. A kernel whose blocks differ in cost states
     * `OmpSchedule::Dynamic`: each thread of the team then takes the next block that no thread has taken. An example
     * is a launch of one block for each tile of a domain whose tiles hold different amounts of work. The other
     * executors ignore the schedule.
     *
     * The type is not in a namespace `omp`: the header of OpenMP declares a namespace of that name for C++.
     */
    enum class OmpSchedule
    {
        Static,
        Dynamic
    };

    namespace trait
    {
        /// The schedule that a kernel states, or the static schedule.
        template<typename T_KernelFn>
        struct OmpScheduleOf : std::integral_constant<OmpSchedule, OmpSchedule::Static>
        {
        };

        template<typename T_KernelFn>
        requires requires { T_KernelFn::ompSchedule; }
        struct OmpScheduleOf<T_KernelFn> : std::integral_constant<OmpSchedule, T_KernelFn::ompSchedule>
        {
        };
    } // namespace trait

    /// The schedule of the blocks of a kernel on the OpenMP executor.
    template<typename T_KernelFn>
    constexpr OmpSchedule ompScheduleOf = trait::OmpScheduleOf<std::remove_cvref_t<T_KernelFn>>::value;
} // namespace alpaka
