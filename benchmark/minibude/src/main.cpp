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
#include <sstream>
#include <stdexcept>
#include <string>
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
            a.wgsizeValues = parsePositiveListOrDefault(raw, 256u, "wgsize");
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
        a.wgsizeValues = {256u};
    return a;
}

struct TimingStats
{
    double min;
    double avg;
    double max;
};

static TimingStats computeStats(std::vector<double> const& values)
{
    if(values.empty())
        return TimingStats{0.0, 0.0, 0.0};

    double mn = std::numeric_limits<double>::infinity();
    double mx = 0.0;
    double sum = 0.0;
    for(double v : values)
    {
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        sum += v;
    }
    double avg = sum / static_cast<double>(values.size());
    return TimingStats{mn, avg, mx};
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

            struct ComboMetrics
            {
                std::uint32_t ppwi = 0u;
                std::uint32_t requestedWg = 0u;
                std::uint32_t actualWg = 0u;
                TimingStats kernelStats{};
                TimingStats contextStats{};
            };

            std::vector<ComboMetrics> combos;
            combos.reserve(args.ppwiValues.size() * args.wgsizeValues.size());

            struct CachedCombo
            {
                TimingStats kernelStats{};
                TimingStats contextStats{};
                std::vector<float> energies;
                std::uint32_t actualWg = 0u;
            };

            using ComboKey = std::pair<std::uint32_t, std::uint32_t>;
            std::map<ComboKey, CachedCombo> comboCache;
            std::vector<std::pair<std::uint32_t, std::uint32_t>> clampedWgsizeNotes;

            ComboMetrics bestMetrics{};
            TimingStats bestKernelStats{};
            TimingStats bestContextStats{};
            bool hasBest = false;
            std::vector<float> bestEnergies;

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

            bool const autotune = (args.ppwiValues.size() * args.wgsizeValues.size() > 1);
            constexpr bool noWorkgroupBackend = decltype(context)::kIsNoWorkgroupBackend;

            for(std::uint32_t ppwi : args.ppwiValues)
            {
                for(std::uint32_t wgsize : args.wgsizeValues)
                {
                    ComboMetrics metrics{};
                    metrics.ppwi = ppwi;
                    metrics.requestedWg = wgsize;

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
                            context.copy_energies(newEntry.energies);

                            cacheIt = comboCache.emplace(key, std::move(newEntry)).first;
                        }

                        cacheEntry = &cacheIt->second;
                        metrics.actualWg = cacheEntry->actualWg;
                        metrics.kernelStats = cacheEntry->kernelStats;
                        metrics.contextStats = cacheEntry->contextStats;

                        if(noWorkgroupBackend && metrics.requestedWg != metrics.actualWg)
                        {
                            auto const clamp = std::make_pair(metrics.requestedWg, metrics.actualWg);
                            if(std::find(clampedWgsizeNotes.begin(), clampedWgsizeNotes.end(), clamp)
                                == clampedWgsizeNotes.end())
                            {
                                clampedWgsizeNotes.push_back(clamp);
                            }
                        }
                    });

                    combos.push_back(metrics);

                    if(cacheEntry != nullptr)
                    {
                        if(!hasBest || better(metrics, metrics.kernelStats, bestMetrics, bestKernelStats))
                        {
                            hasBest = true;
                            bestMetrics = metrics;
                            bestKernelStats = metrics.kernelStats;
                            bestContextStats = metrics.contextStats;
                            bestEnergies = cacheEntry->energies;
                        }
                    }
                }
            }

            if(!hasBest)
                throw std::runtime_error("No valid configuration evaluated for miniBUDE");

            VerifyResult verifyRes{};
            bool const doVerify = args.verify;
            if(doVerify)
            {
                if(refE.size() != bestEnergies.size())
                {
                    std::cerr << "# WARNING: ref_energies count (" << refE.size() << ") != poses count ("
                              << bestEnergies.size() << ")\n";
                }
                verifyRes = verifyEnergies(refE, bestEnergies, args.tol_pct, args.rows, args.csv);
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
            constexpr bool isCpuSerialBackend = decltype(context)::kIsCpuSerialBackend;
            auto const backendTypeLabel = isCpuSerialBackend ? "sequential" : "parallel";

            if(args.csv)
            {
                if(!firstSection)
                    std::cout << '\n';
                std::cout << std::fixed << std::setprecision(3);
                std::cout << "metric,min_ms,avg_ms,max_ms,accelerator,device,data,runs,ppwi,wgsize\n";
                auto emitRow = [&](std::string const& metric, TimingStats const& stats, ComboMetrics const& metrics)
                {
                    std::cout << metric << ',' << stats.min << ',' << stats.avg << ',' << stats.max << ','
                              << acceleratorName << ',' << deviceName << ',' << dataSummaryCsv << ',' << args.runs
                              << ',' << metrics.ppwi << ',' << metrics.actualWg << '\n';
                };
                for(auto const& metrics : combos)
                {
                    emitRow("kernel", metrics.kernelStats, metrics);
                    emitRow("context_d2h", metrics.contextStats, metrics);
                }
                emitRow("best", bestKernelStats, bestMetrics);
                if(doVerify)
                {
                    std::cout << "verify," << (verifyRes.valid ? 1.0 : 0.0) << ',' << verifyRes.max_diff_pct << ','
                              << args.tol_pct << ',' << acceleratorName << ',' << deviceName << ',' << dataSummaryCsv
                              << ',' << args.runs << ',' << bestMetrics.ppwi << ',' << bestMetrics.actualWg << '\n';
                }
            }
            else
            {
                if(!firstSection)
                    std::cout << '\n';

                std::cout << "Accelerator:" << acceleratorName << '\n';
                std::cout << "BackendType:" << backendTypeLabel << '\n';
                std::cout << "Device:" << deviceName << '\n';
                std::cout << "Data: " << dataSummaryHuman << '\n';
                std::cout << "Runs:" << args.runs << '\n';

                if(autotune)
                {
                    std::cout << std::fixed << std::setprecision(3);
                    std::cout << std::left << std::setw(16) << "Config" << std::right << std::setw(10) << "Min(ms)"
                              << std::setw(10) << "Max(ms)" << std::setw(10) << "Avg(ms)" << '\n';
                    for(auto const& metrics : combos)
                    {
                        std::ostringstream label;
                        label << "ppwi=" << metrics.ppwi << ",wg=" << metrics.actualWg;
                        if(metrics.requestedWg != metrics.actualWg)
                            label << " (req=" << metrics.requestedWg << ')';
                        std::cout << std::left << std::setw(16) << label.str() << std::right << std::setw(10)
                                  << metrics.kernelStats.min << std::setw(10) << metrics.kernelStats.max
                                  << std::setw(10) << metrics.kernelStats.avg << '\n';
                    }
                    std::cout << "Best: ppwi=" << bestMetrics.ppwi << ", wgsize=" << bestMetrics.actualWg
                              << ", min_ms=" << bestKernelStats.min << ", avg_ms=" << bestKernelStats.avg
                              << ", max_ms=" << bestKernelStats.max << "\n";
                    std::cout << std::defaultfloat;
                }
                else
                {
                    std::cout << std::fixed << std::setprecision(3)
                              << std::left << std::setw(12) << "Kernel"
                              << std::right << std::setw(10) << "Min(ms)" << std::setw(10) << "Max(ms)"
                              << std::setw(10) << "Avg(ms)" << std::setw(8) << "Note" << '\n';
                    std::cout << std::left << std::setw(12) << "Compute" << std::right << std::setw(10)
                              << bestKernelStats.min << std::setw(10) << bestKernelStats.max << std::setw(10)
                              << bestKernelStats.avg << std::setw(8) << "" << '\n';
                    std::cout << std::left << std::setw(12) << "D2H" << std::right << std::setw(10)
                              << bestContextStats.min << std::setw(10) << bestContextStats.max << std::setw(10)
                              << bestContextStats.avg << std::setw(8) << "context" << '\n';
                    std::cout << "context_init_ms: " << init_context_ms << '\n';
                    std::cout << std::defaultfloat;
                }

                if(doVerify)
                {
                    std::cout << "verify: { valid: " << (verifyRes.valid ? "true" : "false")
                              << ", max_diff_%: " << std::setprecision(6) << verifyRes.max_diff_pct
                              << ", tol_%: " << std::setprecision(6) << args.tol_pct << " }\n";
                }

                if(!clampedWgsizeNotes.empty())
                {
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
