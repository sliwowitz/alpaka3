/* Copyright 2026 Jiří Vyskočil
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/mem/IdxRange.hpp"
#include "alpaka/mem/ThreadSpace.hpp"
#include "alpaka/onAcc/layout.hpp"
#include "alpaka/utility.hpp"

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <type_traits>

namespace alpaka::onAcc
{
    /** The runs of a one-dimensional index range that one worker takes.
     *
     * A run is a range of indices of the worker that follow one another in the range with the stride of the range,
     * given as an alpaka::IdxRange. The worker takes the indices that the same layout gives it one by one. With the
     * contiguous layout these indices are one run: the part of the range of the worker. With the strided layout they
     * are runs of one index each, the count of the workers apart. A code that is written for runs, such as a loop
     * that decodes its first index once, walks the part of a worker of the host in one loop, which a compiler can
     * vectorize, and sets up once for the part instead of once for each index.
     */
    template<
        alpaka::concepts::IdxRange T_IdxRange,
        typename T_ThreadSpace,
        typename T_IdxMapperFn,
        alpaka::concepts::CVector T_CSelect>
    class RunIdxContainer
    {
    public:
        using IdxType = typename T_IdxRange::IdxType;
        using IdxVecType = Vec<IdxType, 1u>;
        using RunType = IdxRange<IdxVecType, IdxVecType, IdxVecType>;

        static_assert(T_IdxRange::dim() == 1u, "the runs of a range are defined for a one-dimensional range");

        ALPAKA_FN_ACC constexpr RunIdxContainer(
            T_IdxRange const& idxRange,
            T_ThreadSpace const& threadSpace,
            T_IdxMapperFn idxMapping,
            T_CSelect const& = T_CSelect{})
            : m_idxRange(idxRange)
            , m_threadSpace{threadSpace}
        {
            alpaka::unused(idxMapping);
        }

        /** The runs of the worker in their order.
         *
         * The iterator counts in logical indices, the units of the stride from the begin of the range: the first
         * logical index of the current run, the count of logical indices in a run, the step to the next run of the
         * worker, and the end of the range.
         */
        class ConstIterator
        {
            friend class RunIdxContainer;

            ALPAKA_FN_ACC constexpr ConstIterator(
                IdxType begin,
                IdxType stride,
                IdxType extent,
                IdxType current,
                IdxType length,
                IdxType step,
                IdxType logicalEnd)
                : m_begin{begin}
                , m_stride{stride}
                , m_extent{extent}
                , m_current{current}
                , m_length{length}
                , m_step{step}
                , m_logicalEnd{logicalEnd}
            {
            }

        public:
            using iterator_concept = std::forward_iterator_tag;
            using value_type = RunType;
            using difference_type = std::ptrdiff_t;

            constexpr ConstIterator() = default;

            ALPAKA_FN_ACC constexpr value_type operator*() const
            {
                IdxType const lastLogical = m_current + m_length < m_logicalEnd ? m_current + m_length : m_logicalEnd;
                IdxType const endOffset = lastLogical * m_stride < m_extent ? lastLogical * m_stride : m_extent;
                return value_type{
                    IdxVecType{m_begin + m_current * m_stride},
                    IdxVecType{m_begin + endOffset},
                    IdxVecType{m_stride}};
            }

            ALPAKA_FN_ACC constexpr ConstIterator& operator++()
            {
                m_current += m_step;
                return *this;
            }

            ALPAKA_FN_ACC constexpr ConstIterator operator++(int)
            {
                ConstIterator old = *this;
                ++*this;
                return old;
            }

            /// Two iterators past their last run are equal; otherwise two iterators are equal at the same run.
            ALPAKA_FN_ACC constexpr bool operator==(ConstIterator const& other) const
            {
                if(done() || other.done())
                    return done() == other.done();
                return m_current == other.m_current;
            }

            ALPAKA_FN_ACC constexpr bool operator!=(ConstIterator const& other) const
            {
                return !(*this == other);
            }

        private:
            ALPAKA_FN_ACC constexpr bool done() const
            {
                return m_current >= m_logicalEnd;
            }

            IdxType m_begin = 0;
            IdxType m_stride = 1;
            IdxType m_extent = 0;
            IdxType m_current = 0;
            IdxType m_length = 0;
            IdxType m_step = 1;
            IdxType m_logicalEnd = 0;
        };

        ALPAKA_FN_ACC constexpr ConstIterator begin() const
        {
            auto [threadIdx, numThreads] = m_threadSpace.mapTo(T_CSelect{});
            IdxType const worker = static_cast<IdxType>(threadIdx.x());
            IdxType const workers = static_cast<IdxType>(numThreads.x());
            IdxType const begin = m_idxRange.getBeginMd().x();
            IdxType const stride = m_idxRange.getStrideMd().x();
            IdxType const extent = m_idxRange.distance().x();
            IdxType const logicalExtent = divCeil(extent, stride);

            if constexpr(std::is_same_v<T_IdxMapperFn, layout::Strided>)
            {
                // Runs of one index, the count of the workers apart.
                return ConstIterator{begin, stride, extent, worker, IdxType{1}, workers, logicalExtent};
            }
            else
            {
                static_assert(
                    std::is_same_v<T_IdxMapperFn, layout::Contiguous>,
                    "a range of runs takes the strided or the contiguous layout");
                /* One run: the part of the worker, as the contiguous layout of a tiled traversal gives it. A worker
                 * with an empty part has no run. */
                IdxType const base = logicalExtent / workers;
                IdxType const rem = logicalExtent % workers;
                IdxType const first = worker * base + (worker < rem ? worker : rem);
                IdxType const next = worker + IdxType{1};
                IdxType const last = next * base + (next < rem ? next : rem);
                return ConstIterator{
                    begin,
                    stride,
                    extent,
                    first < last ? first : logicalExtent,
                    last - first,
                    logicalExtent,
                    logicalExtent};
            }
        }

        ALPAKA_FN_ACC constexpr ConstIterator end() const
        {
            IdxType const stride = m_idxRange.getStrideMd().x();
            IdxType const extent = m_idxRange.distance().x();
            IdxType const logicalExtent = divCeil(extent, stride);
            return ConstIterator{
                m_idxRange.getBeginMd().x(),
                stride,
                extent,
                logicalExtent,
                IdxType{0},
                IdxType{1},
                logicalExtent};
        }

    private:
        T_IdxRange m_idxRange;
        T_ThreadSpace m_threadSpace;
    };
} // namespace alpaka::onAcc
