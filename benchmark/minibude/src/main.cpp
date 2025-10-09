// SPDX-License-Identifier: MPL-2.0
// Portions adapted conceptually from UoB-HPC/miniBUDE (Apache-2.0).
// Author: Ivan Andriievskyi, Jiří Vyskočil
// Work funded by US NAS and ONRG (IMPRESS-U).

#include "minibude.hpp"
#include "physics_minibude.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <alpaka/onHost/demangledName.hpp>

namespace onHost = alpaka::onHost;

static std::vector<std::uint32_t> parsePositiveListOrDefault(
    std::string const& text,
    std::uint32_t defaultValue,
    std::string const& optionName)
{
    if(text.empty())
        return {defaultValue};

    std::vector<std::uint32_t> values;
    std::stringstream ss(text);
    std::string token;
    while(std::getline(ss, token, ','))
    {
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char ch) { return std::isspace(ch); }), token.end());
        if(token.empty())
            continue;

        std::size_t idx = 0;
        unsigned long parsed = 0;
        try
        {
            parsed = std::stoul(token, &idx, 10);
        }
        catch(std::exception const&)
        {
            throw std::runtime_error("Invalid value '" + token + "' for option --" + optionName + "");
        }
        if(idx != token.size())
            throw std::runtime_error("Invalid value '" + token + "' for option --" + optionName + "");
        if(parsed == 0ul)
            throw std::runtime_error("Option --" + optionName + " requires positive integers");
        if(parsed > std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("Option --" + optionName + " value out of range");
        values.push_back(static_cast<std::uint32_t>(parsed));
    }

    if(values.empty())
        throw std::runtime_error("Option --" + optionName + " did not contain valid values");

    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

struct Args
{
    std::string deckDir;
    int runs = 1;
    bool verify = false;
    double tol_pct = 0.025;
    std::size_t rows = 8;
    std::string dump_path;
    bool csv = false;
    std::vector<std::uint32_t> ppwiValues;
    std::vector<std::uint32_t> wgsizeValues;
    std::optional<std::size_t> posesLimit;
};

static Args parseArgs(int argc, char** argv)
{
    Args a;
    for(int i = 1; i < argc; ++i)
    {
        std::string s(argv[i]);
        auto next_str = [&](std::string& dst)
        {
            if(i + 1 < argc)
                dst = argv[++i];
        };
        auto next_i = [&](int& dst)
        {
            if(i + 1 < argc)
                dst = std::stoi(argv[++i]);
        };
        auto next_z = [&](std::size_t& dst)
        {
            if(i + 1 < argc)
                dst = std::stoul(argv[++i]);
        };
        auto next_d = [&](double& dst)
        {
            if(i + 1 < argc)
                dst = std::stod(argv[++i]);
        };

        if(s == "--deck")
            next_str(a.deckDir);
        else if(s == "--runs")
            next_i(a.runs);
        else if(s == "--verify")
            a.verify = true;
        else if(s == "--tol-pct")
            next_d(a.tol_pct);
        else if(s == "--rows")
            next_z(a.rows);
        else if(s == "--dump-energies")
            next_str(a.dump_path);
        else if(s == "--csv")
            a.csv = true;
        else if(s == "--poses" || s == "-n")
        {
            std::size_t limit = 0;
            next_z(limit);
            a.posesLimit = limit;
        }
        else if(s == "--ppwi")
        {
            std::string raw;
            next_str(raw);
            if(raw.empty())
                throw std::runtime_error("--ppwi requires a comma separated list");
            a.ppwiValues = parsePositiveListOrDefault(raw, 1u, "ppwi");
        }
        else if(s == "--wgsize")
        {
            std::string raw;
            next_str(raw);
            if(raw.empty())
                throw std::runtime_error("--wgsize requires a comma separated list");
            a.wgsizeValues = parsePositiveListOrDefault(raw, 1u, "wgsize");
        }
        else if(s == "-h" || s == "--help")
        {
            std::cout << "miniBUDE (CPU, deck mode)\n"
                      << "  --deck <DIR>        directory with bm1/{ligand,protein,forcefield,poses,ref_energies}\n"
                      << "  --runs <R>          timed repeats (default " << a.runs << ")\n"
                      << "  --verify            compare against ref_energies.out\n"
                      << "  --tol-pct <pct>     tolerance percent (default " << a.tol_pct << ")\n"
                      << "  --rows <N>          number of mismatches to print (default " << a.rows << ")\n"
                      << "  --dump-energies P   write computed energies to file P\n"
                      << "  --ppwi v1[,v2,...]  poses per work-item list (default 1)\n"
                      << "  --wgsize v1[,v2,...] workgroup size list (default 256)\n"
                      << "  --csv               CSV summary\n";
            std::exit(0);
        }
    }
    if(a.deckDir.empty())
    {
        throw std::runtime_error("Deck path is required. Use --deck <DIR>");
    }
    if(a.runs < 1)
        a.runs = 1;
    if(a.ppwiValues.empty())
        a.ppwiValues = {1u};
    if(a.wgsizeValues.empty())
        a.wgsizeValues = {1u};
    return a;
}

struct TimingStats
{
    double sum;
    double avg;
    double min;
    double max;
    double stddev;
};

static TimingStats computeStats(std::vector<double> const& values)
{
    if(values.empty())
        return TimingStats{0.0, 0.0, 0.0, 0.0, 0.0};

    double const sum = std::accumulate(values.begin(), values.end(), 0.0);
    double const avg = sum / static_cast<double>(values.size());
    auto const [mnIt, mxIt] = std::minmax_element(values.begin(), values.end());
    double variance = 0.0;
    for(double v : values)
    {
        double const diff = v - avg;
        variance += diff * diff;
    }
    variance /= static_cast<double>(values.size());
    double const stddev = std::sqrt(variance);
    return TimingStats{sum, avg, *mnIt, *mxIt, stddev};
}

int main(int argc, char** argv)
{
    auto const args = parseArgs(argc, argv);
    std::string const& D = args.deckDir;

    // Read deck
    auto const ligand = readBinaryVec<DeckAtom>(D + "/ligand.in");
    auto const protein = readBinaryVec<DeckAtom>(D + "/protein.in");
    auto const ff = readBinaryVec<FFParams>(D + "/forcefield.in");

    std::array<std::vector<float>, 6> dof;
    {
        auto const flat = readBinaryVec<float>(D + "/poses.in");
        splitPoses6(flat, dof); // rx,ry,rz,tx,ty,tz
        std::size_t const totalPoses = dof[0].size();
        if(args.posesLimit)
        {
            if(*args.posesLimit == 0u)
                throw std::runtime_error("--poses value must be > 0");
            if(*args.posesLimit > totalPoses)
            {
                throw std::runtime_error(
                    "--poses value " + std::to_string(*args.posesLimit)
                    + " exceeds deck pose count " + std::to_string(totalPoses));
            }
            for(auto& component : dof)
                component.resize(*args.posesLimit);
        }
    }

    std::vector<float> refE;
    if(args.verify)
        refE = readRefEnergiesTxt(D + "/ref_energies.out");

    float volatile sink = 0.0f;
    bool anyBackend = false;
    bool firstSection = true;

    auto const backends = onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors);
    (void) onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        {
            anyBackend = true;

            for(auto v : args.ppwiValues)
            {
                if(!isSupportedPpwi(v))
                    throw std::runtime_error("Unsupported PPWI value " + std::to_string(v));
            }

            MiniBudeContext<decltype(backend)> context(backend, ligand, protein, ff, dof);
            double const init_context_ms = context.init(ligand, protein, ff, dof);

            struct CachedCombo;

            struct ComboMetrics
            {
                std::uint32_t ppwi = 0u;
                std::uint32_t requestedWg = 0u;
                std::uint32_t actualWg = 0u;
                TimingStats kernelStats{};
                TimingStats contextStats{};
                CachedCombo const* cache = nullptr;
            };

            std::vector<ComboMetrics> combos;
            combos.reserve(args.ppwiValues.size() * args.wgsizeValues.size());

            struct CachedCombo
            {
                TimingStats kernelStats{};
                TimingStats contextStats{};
                std::vector<float> energies;
                std::uint32_t actualWg = 0u;
                std::vector<double> kernelSamples;
                std::optional<VerifyResult> verify;
            };

            using ComboKey = std::pair<std::uint32_t, std::uint32_t>;
            std::map<ComboKey, CachedCombo> comboCache;
            std::vector<std::pair<std::uint32_t, std::uint32_t>> clampedWgsizeNotes;
            std::set<ComboKey> printedCombos;

            ComboMetrics bestMetrics{};
            TimingStats bestKernelStats{};
            bool hasBest = false;
            std::vector<float> bestEnergies;
            std::optional<VerifyResult> bestVerify;

            auto better = [](ComboMetrics const& lhs, TimingStats const& lhsKernel, ComboMetrics const& rhs, TimingStats const& rhsKernel) {
                constexpr double eps = 1e-9;
                auto cmp = [eps](double a, double b) {
                    if(a < b - eps)
                        return -1;
                    if(a > b + eps)
                        return 1;
                    return 0;
                };
                int const cmpMin = cmp(lhsKernel.min, rhsKernel.min);
                if(cmpMin < 0)
                    return true;
                if(cmpMin > 0)
                    return false;
                int const cmpAvg = cmp(lhsKernel.avg, rhsKernel.avg);
                if(cmpAvg < 0)
                    return true;
                if(cmpAvg > 0)
                    return false;
                if(lhs.ppwi < rhs.ppwi)
                    return true;
                if(lhs.ppwi > rhs.ppwi)
                    return false;
                return lhs.actualWg < rhs.actualWg;
            };

            constexpr bool noWorkgroupBackend
                = alpaka::exec::isSeqExecutor_v<typename decltype(context)::ExecType>;

            for(std::uint32_t ppwi : args.ppwiValues)
            {
                for(std::uint32_t wgsize : args.wgsizeValues)
                {
                    ComboMetrics metrics{};
                    metrics.ppwi = ppwi;
                    metrics.requestedWg = wgsize;

                    bool inserted = false;
                    CachedCombo const* cacheEntry = nullptr;

                    dispatchPpwi(ppwi, [&](auto ppwiConst) {
                        constexpr std::uint32_t PPWI = ppwiConst;
                        auto const specInfo = context.template makeThreadSpec<PPWI>(wgsize);
                        ComboKey const key{PPWI, specInfo.actualWgsize};

                        auto cacheIt = comboCache.find(key);
                        if(cacheIt == comboCache.end())
                        {
                            std::vector<double> kernel_ms;
                            std::vector<double> context_ms;
                            kernel_ms.reserve(static_cast<std::size_t>(args.runs));
                            context_ms.reserve(static_cast<std::size_t>(args.runs));

                            context.template run<PPWI>(specInfo.spec);
                            sink += context.accumulate_output();

                            for(int r = 0; r < args.runs; ++r)
                            {
                                auto const timings = context.template run<PPWI>(specInfo.spec);
                                sink += context.accumulate_output();
                                kernel_ms.push_back(timings.kernel_ms);
                                context_ms.push_back(timings.d2h_ms);
                            }

                            CachedCombo newEntry{};
                            newEntry.kernelStats = computeStats(kernel_ms);
                            newEntry.contextStats = computeStats(context_ms);
                            newEntry.actualWg = specInfo.actualWgsize;
                            newEntry.kernelSamples = std::move(kernel_ms);
                            context.copy_energies(newEntry.energies);
                            if(args.verify)
                                newEntry.verify = verifyEnergies(refE, newEntry.energies, args.tol_pct, args.rows, args.csv);

                            cacheIt = comboCache.emplace(key, std::move(newEntry)).first;
                            inserted = true;
                        }

                        cacheEntry = &cacheIt->second;

                        if(noWorkgroupBackend && specInfo.actualWgsize != wgsize)
                        {
                            auto const clamp = std::make_pair(wgsize, specInfo.actualWgsize);
                            if(std::find(clampedWgsizeNotes.begin(), clampedWgsizeNotes.end(), clamp)
                                == clampedWgsizeNotes.end())
                            {
                                clampedWgsizeNotes.push_back(clamp);
                            }
                        }

                        if(inserted)
                        {
                            metrics.actualWg = cacheEntry->actualWg;
                            metrics.kernelStats = cacheEntry->kernelStats;
                            metrics.contextStats = cacheEntry->contextStats;
                            metrics.cache = cacheEntry;
                        }
                    });

                    if(inserted && cacheEntry != nullptr)
                    {
                        combos.push_back(metrics);

                        if(!hasBest || better(metrics, metrics.kernelStats, bestMetrics, bestKernelStats))
                        {
                            hasBest = true;
                            bestMetrics = metrics;
                            bestKernelStats = metrics.kernelStats;
                            bestEnergies = cacheEntry->energies;
                            bestVerify = cacheEntry->verify;
                        }
                    }
                }
            }

            if(!hasBest)
                throw std::runtime_error("No valid configuration evaluated for miniBUDE");

            bool const doVerify = args.verify;
            bool warnPoseCountMismatch = false;
            if(doVerify)
            {
                if(!bestVerify.has_value())
                    bestVerify = verifyEnergies(refE, bestEnergies, args.tol_pct, args.rows, args.csv);
                warnPoseCountMismatch = (refE.size() != bestEnergies.size());
                if(warnPoseCountMismatch && args.csv)
                {
                    std::cerr << "# WARNING: ref_energies count (" << refE.size() << ") != poses count ("
                              << bestEnergies.size() << ")\n";
                }
            }

            if(!args.dump_path.empty())
            {
                std::ofstream out(args.dump_path, std::ios::out | std::ios::trunc);
                for(float e : bestEnergies)
                    out << std::setprecision(7) << e << '\n';
                if(!args.csv)
                {
                    std::cout << (out ? "# energies dumped to " + args.dump_path + "\n"
                                      : "# ERROR: failed to write " + args.dump_path + "\n");
                }
            }

            auto const acceleratorName = alpaka::onHost::demangledName(context.exec);
            auto const deviceName = context.device.getName();
            auto const dataSummaryHuman = "poses=" + std::to_string(context.nposes) + ", ligand="
                + std::to_string(context.natlig) + ", protein=" + std::to_string(context.natpro)
                + ", ff=" + std::to_string(context.ffCount);
            auto const dataSummaryCsv = "poses=" + std::to_string(context.nposes) + ";ligand="
                + std::to_string(context.natlig) + ";protein=" + std::to_string(context.natpro)
                + ";ff=" + std::to_string(context.ffCount);

            double const interactions = static_cast<double>(context.nposes) * context.natlig * context.natpro;

            std::vector<ComboMetrics const*> uniqueCombos;
            uniqueCombos.reserve(combos.size());
            for(auto const& metrics : combos)
            {
                ComboKey const key{metrics.ppwi, metrics.actualWg};
                if(printedCombos.insert(key).second)
                    uniqueCombos.push_back(&metrics);
            }

            auto throughputFor = [&](ComboMetrics const& metrics) {
                if(metrics.kernelStats.min <= 0.0)
                    return std::tuple<double, double, double>{0.0, 0.0, 0.0};
                double const seconds = metrics.kernelStats.min / 1000.0;
                double const gigaInteractionsPerSec = (seconds > 0.0) ? (interactions / seconds) / 1e9 : 0.0;
                double const gflops = gigaInteractionsPerSec * kFlopsPerInteraction;
                double const gfinsts = gigaInteractionsPerSec * kInstsPerInteraction;
                return std::tuple<double, double, double>{gigaInteractionsPerSec, gflops, gfinsts};
            };

            if(args.csv)
            {
                if(!firstSection)
                    std::cout << '\n';
                std::cout << std::fixed << std::setprecision(3);
                std::cout << "metric,sum_ms,avg_ms,min_ms,max_ms,stddev_ms,giga_interactions_s,gflop_s,gfinst_s,"
                             "accelerator,device,data,runs,ppwi,wgsize";
#ifdef MINIBUDE_ENABLE_CSV_BUILD_TAG
                std::cout << ",build_flags_tag";
#endif
#ifdef MINIBUDE_ENABLE_CSV_KERNEL_TAG
                std::cout << ",kernel_path_tag";
#endif
                std::cout << '\n';
                auto emitRow = [&](
                                      std::string const& metric,
                                      TimingStats const& stats,
                                      ComboMetrics const& metrics,
                                      double gigaInteractions,
                                      double gflops,
                                      double gfinsts)
                {
                    std::cout << metric << ',' << stats.sum << ',' << stats.avg << ',' << stats.min << ','
                              << stats.max << ',' << stats.stddev << ',' << gigaInteractions << ',' << gflops << ','
                              << gfinsts << ',' << acceleratorName << ',' << deviceName << ',' << dataSummaryCsv
                              << ',' << args.runs << ',' << metrics.ppwi << ',' << metrics.actualWg;
#ifdef MINIBUDE_ENABLE_CSV_BUILD_TAG
                    std::cout << ',' << MINIBUDE_BUILD_FLAGS_TAG;
#endif
#ifdef MINIBUDE_ENABLE_CSV_KERNEL_TAG
                    std::cout << ',' << MINIBUDE_KERNEL_PATH_TAG;
#endif
                    std::cout << '\n';
                };
                for(auto const* metricsPtr : uniqueCombos)
                {
                    auto const& metrics = *metricsPtr;
                    auto const [gInteractionsPerSec, gflops, gfinsts] = throughputFor(metrics);
                    emitRow("kernel", metrics.kernelStats, metrics, gInteractionsPerSec, gflops, gfinsts);
                    emitRow("context_d2h", metrics.contextStats, metrics, 0.0, 0.0, 0.0);
                }
                auto const [bestGInteractions, bestGflops, bestGfinsts] = throughputFor(bestMetrics);
                emitRow("best", bestKernelStats, bestMetrics, bestGInteractions, bestGflops, bestGfinsts);
                if(doVerify && bestVerify)
                {
                    std::cout << "verify," << (bestVerify->valid ? 1.0 : 0.0) << ',' << bestVerify->max_diff_pct
                              << ',' << args.tol_pct << ',' << acceleratorName << ',' << deviceName << ','
                              << dataSummaryCsv << ',' << args.runs << ',' << bestMetrics.ppwi << ','
                              << bestMetrics.actualWg;
#ifdef MINIBUDE_ENABLE_CSV_BUILD_TAG
                    std::cout << ',' << MINIBUDE_BUILD_FLAGS_TAG;
#endif
#ifdef MINIBUDE_ENABLE_CSV_KERNEL_TAG
                    std::cout << ',' << MINIBUDE_KERNEL_PATH_TAG;
#endif
                    std::cout << '\n';
                }
                std::cout << std::defaultfloat;
            }
            else
            {
                if(!firstSection)
                    std::cout << '\n';

                std::cout << "Accelerator:" << acceleratorName << '\n';
                std::cout << "Device:" << deviceName << '\n';
                std::cout << "Data: " << dataSummaryHuman << '\n';
                std::cout << "Runs:" << args.runs << '\n';

                if(warnPoseCountMismatch)
                {
                    std::cout << "# WARNING: ref_energies count (" << refE.size() << ") != poses count ("
                              << bestEnergies.size() << ")\n";
                }

                std::cout << "results:\n";
                for(auto const* metricsPtr : uniqueCombos)
                {
                    auto const& metrics = *metricsPtr;
                    auto const* cacheEntry = metrics.cache;
                    if(cacheEntry == nullptr)
                        continue;
                    auto const [gInteractionsPerSec, gflops, gfinsts] = throughputFor(metrics);
                    auto const& kernelSamples = cacheEntry->kernelSamples;

                    auto formatSamples = [](std::vector<double> const& samples) {
                        std::ostringstream oss;
                        oss << std::fixed << std::setprecision(3);
                        for(std::size_t i = 0; i < samples.size(); ++i)
                        {
                            if(i != 0u)
                                oss << ',';
                            oss << samples[i];
                        }
                        return oss.str();
                    };

                    auto const& energies = cacheEntry->energies;
                    std::size_t const energiesToPrint = std::min<std::size_t>(8u, energies.size());
                    double const contextMs = init_context_ms;

                    bool const valid = doVerify ? (cacheEntry->verify ? cacheEntry->verify->valid : false) : true;
                    double const maxDiffPct
                        = doVerify ? (cacheEntry->verify ? cacheEntry->verify->max_diff_pct : 0.0) : 0.0;
                    std::ostringstream maxDiffStr;
                    maxDiffStr << std::fixed << std::setprecision(6) << maxDiffPct;

                    std::cout << std::fixed << std::setprecision(3);
                    std::cout << "  - outcome:             { valid: " << (valid ? "true" : "false")
                              << ", max_diff_%: " << maxDiffStr.str() << " }\n";
                    std::cout << "    param:               { ppwi: " << metrics.ppwi << ", wgsize: " << metrics.actualWg
                              << " }\n";
                    std::cout << "    raw_iterations:      [" << formatSamples(kernelSamples) << "]\n";
                    std::cout << "    context_ms:          " << contextMs << "\n";
                    std::cout << "    sum_ms:              " << metrics.kernelStats.sum << "\n";
                    std::cout << "    avg_ms:              " << metrics.kernelStats.avg << "\n";
                    std::cout << "    min_ms:              " << metrics.kernelStats.min << "\n";
                    std::cout << "    max_ms:              " << metrics.kernelStats.max << "\n";
                    std::cout << "    stddev_ms:           " << metrics.kernelStats.stddev << "\n";
                    std::cout << "    giga_interactions/s: " << gInteractionsPerSec << "\n";
                    std::cout << "    gflop/s:             " << gflops << "\n";
                    std::cout << "    gfinst/s:            " << gfinsts << "\n";
                    std::cout << "    energies:            \n";
                    std::cout << std::setprecision(2);
                    for(std::size_t i = 0; i < energiesToPrint; ++i)
                        std::cout << "      - " << energies[i] << '\n';
                    std::cout << std::defaultfloat;
                }

                std::cout << std::fixed << std::setprecision(2);
                std::cout << "best: { min_ms: " << bestKernelStats.min << ", max_ms: " << bestKernelStats.max
                          << ", sum_ms: " << bestKernelStats.sum << ", avg_ms: " << bestKernelStats.avg
                          << ", ppwi: " << bestMetrics.ppwi << ", wgsize: " << bestMetrics.actualWg << " }\n";
                std::cout << std::defaultfloat;

                if(!clampedWgsizeNotes.empty())
                {
                    std::sort(clampedWgsizeNotes.begin(), clampedWgsizeNotes.end());
                    std::ostringstream note;
                    note << "# note: requested wgsize values clamped by backend: ";
                    for(std::size_t i = 0; i < clampedWgsizeNotes.size(); ++i)
                    {
                        if(i != 0u)
                            note << ", ";
                        note << clampedWgsizeNotes[i].first << "->" << clampedWgsizeNotes[i].second;
                    }
                    std::cout << note.str() << '\n';
                }
            }

            firstSection = false;
        },
        backends);

    if(!anyBackend)
        throw std::runtime_error("No suitable Alpaka backend found for miniBUDE.");

    (void) sink;
    return 0;
}
