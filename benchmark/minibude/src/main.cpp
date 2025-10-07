// SPDX-License-Identifier: MPL-2.0
// Portions adapted conceptually from UoB-HPC/miniBUDE (Apache-2.0).
// Author: Ivan Andriievskyi, Jiří Vyskočil
// Work funded by US NAS and ONRG (IMPRESS-U).

#include "minibude.hpp"
#include "physics_minibude.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace onHost = alpaka::onHost;

struct Args
{
    std::string deckDir;
    int runs = 1;
    bool verify = false;
    double tol_pct = 0.025;
    std::size_t rows = 8;
    std::string dump_path;
    bool csv = false;
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
        else if(s == "-h" || s == "--help")
        {
            std::cout << "miniBUDE (CPU, deck mode)\n"
                      << "  --deck <DIR>        directory with bm1/{ligand,protein,forcefield,poses,ref_energies}\n"
                      << "  --runs <R>          timed repeats (default " << a.runs << ")\n"
                      << "  --verify            compare against ref_energies.out\n"
                      << "  --tol-pct <pct>     tolerance percent (default " << a.tol_pct << ")\n"
                      << "  --rows <N>          number of mismatches to print (default " << a.rows << ")\n"
                      << "  --dump-energies P   write computed energies to file P\n"
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

    std::cout << "miniBUDE deck mode\n"
              << "  ligand atoms : " << ligand.size() << "\n"
              << "  protein atoms: " << protein.size() << "\n"
              << "  ff params    : " << ff.size() << "\n"
              << "  poses        : " << dof[0].size() << "\n"
              << (args.verify ? "  ref energies : " : "") << (args.verify ? std::to_string(refE.size()) : "")
              << (args.verify ? "\n" : "");
    if(!dof[0].empty())
    {
        std::cout << "  pose[0] = { " << dof[0][0] << ", " << dof[1][0] << ", " << dof[2][0] << ", " << dof[3][0]
                  << ", " << dof[4][0] << ", " << dof[5][0] << " }\n";
    }

    std::size_t const N = dof[0].size();

    // Context-assisted execution
    float volatile sink = 0.0f;
    std::vector<float> energies(N, 0.0f);
    std::vector<double> kernel_ms;
    std::vector<double> context_ms;
    double init_context_ms = 0.0;
    bool executed = false;

    (void) onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        {
            if(executed)
                return;

            MiniBudeContext<decltype(backend)> context(backend, ligand, protein, ff, dof);
            init_context_ms = context.init(ligand, protein, ff, dof);

            // Warmup run (excluded from stats, ensures residency)
            (void) context.run_once();
            sink += context.accumulate_output();

            kernel_ms.resize(static_cast<std::size_t>(args.runs));
            context_ms.resize(static_cast<std::size_t>(args.runs));

            for(int r = 0; r < args.runs; ++r)
            {
                auto const timings = context.run_once();
                std::size_t const idx = static_cast<std::size_t>(r);
                kernel_ms[idx] = timings.kernel_ms;
                context_ms[idx] = timings.d2h_ms;
                sink += context.accumulate_output();
            }

            context.copy_energies(energies);
            executed = true;
        },
        onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors));

    if(!executed)
        throw std::runtime_error("No suitable Alpaka backend found for miniBUDE.");

    auto const kernelStats = computeStats(kernel_ms);
    auto const contextStats = computeStats(context_ms);

    if(args.csv)
    {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "metric,min_ms,avg_ms,max_ms\n";
        std::cout << "kernel," << kernelStats.min << "," << kernelStats.avg << "," << kernelStats.max << "\n";
        std::cout << "context_d2h," << contextStats.min << "," << contextStats.avg << "," << contextStats.max
                  << "\n";
        std::cout << "context_init," << init_context_ms << "," << init_context_ms << "," << init_context_ms
                  << "\n";
    }
    else
    {
        std::cout << std::fixed << std::setprecision(3)
                  << "kernel_ms: min/avg/max = " << kernelStats.min << " / " << kernelStats.avg << " / "
                  << kernelStats.max << "\n"
                  << "context_ms (d2h): min/avg/max = " << contextStats.min << " / " << contextStats.avg << " / "
                  << contextStats.max << "\n"
                  << "context_init_ms: " << init_context_ms << "\n";
    }

    if(!args.dump_path.empty())
    {
        std::ofstream out(args.dump_path, std::ios::out | std::ios::trunc);
        for(float e : energies)
            out << std::setprecision(7) << e << "\n";
        std::cout
            << (out ? "# energies dumped to " + args.dump_path + "\n"
                    : "# ERROR: failed to write " + args.dump_path + "\n");
    }

    if(args.verify)
    {
        if(refE.size() != energies.size())
        {
            std::cerr << "# WARNING: ref_energies count (" << refE.size() << ") != poses count (" << energies.size()
                      << ")\n";
        }
        auto res = verifyEnergies(refE, energies, args.tol_pct, args.rows, args.csv);
        if(args.csv)
        {
            std::cout << "valid,max_diff_pct\n"
                      << (res.valid ? "true" : "false") << "," << std::setprecision(6) << res.max_diff_pct << "\n";
        }
        else
        {
            std::cout << "verify: { valid: " << (res.valid ? "true" : "false")
                      << ", max_diff_%: " << std::setprecision(6) << res.max_diff_pct << ", tol_%: " << args.tol_pct
                      << " }\n";
        }
    }

    (void) sink;
    return 0;
}
