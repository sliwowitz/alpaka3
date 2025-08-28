// SPDX-License-Identifier: Apache-2.0
// Portions adapted conceptually from UoB-HPC/miniBUDE (Apache-2.0).
// Author: Ivan Andriievskyi
// Work funded by US NAS and ONRG (IMPRESS-U).

#include "physics_ref.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

struct Args
{
    std::uint64_t poses = 500000; // number of poses to score
    int runs = 5; // repeats for timing
    std::size_t natlig = 256; // ligand atoms
    std::size_t natpro = 4096; // protein atoms
};

static Args parseArgs(int argc, char** argv)
{
    Args a;
    for(int i = 1; i < argc; ++i)
    {
        std::string s(argv[i]);
        auto getU64 = [&](std::uint64_t& dst)
        {
            if(i + 1 < argc)
                dst = std::stoull(argv[++i]);
        };
        auto getI = [&](int& dst)
        {
            if(i + 1 < argc)
                dst = std::stoi(argv[++i]);
        };
        auto getZ = [&](std::size_t& dst)
        {
            if(i + 1 < argc)
                dst = std::stoul(argv[++i]);
        };
        if(s == "--poses")
            getU64(a.poses);
        else if(s == "--runs")
            getI(a.runs);
        else if(s == "--natlig")
            getZ(a.natlig);
        else if(s == "--natpro")
            getZ(a.natpro);
        else if(s == "-h" || s == "--help")
        {
            std::cout << "miniBUDE (CPU, synthetic deck)\n"
                      << "  --poses  <N>      number of poses (default " << a.poses << ")\n"
                      << "  --runs   <R>      timed repeats (default " << a.runs << ")\n"
                      << "  --natlig <L>      ligand atoms (default " << a.natlig << ")\n"
                      << "  --natpro <P>      protein atoms (default " << a.natpro << ")\n";
            std::exit(0);
        }
    }
    if(a.runs < 1)
        a.runs = 1;
    return a;
}

template<class F>
static double time_ms(F&& f)
{
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    f();
    std::chrono::duration<double, std::milli> dt = clock::now() - t0;
    return dt.count();
}

// Create a synthetic system (positions, charges, LJ params).
static void makeSynthetic(
    std::size_t natlig,
    std::size_t natpro,
    std::vector<Atom>& ligand,
    std::vector<Atom>& protein)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> posL(-2.0f, 2.0f);
    std::uniform_real_distribution<float> posP(0.0f, 100.0f);
    std::uniform_real_distribution<float> qdist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> sdist(2.5f, 4.5f); // sigma ~ 2.5–4.5 Å
    std::uniform_real_distribution<float> edist(0.01f, 0.20f); // epsilon ~ 0.01–0.2

    ligand.resize(natlig);
    for(auto& a : ligand)
    {
        a.x = posL(rng);
        a.y = posL(rng);
        a.z = posL(rng);
        a.q = qdist(rng);
        a.sigma = sdist(rng);
        a.epsilon = edist(rng);
    }

    protein.resize(natpro);
    for(auto& a : protein)
    {
        a.x = posP(rng);
        a.y = posP(rng);
        a.z = posP(rng);
        a.q = qdist(rng);
        a.sigma = sdist(rng);
        a.epsilon = edist(rng);
    }
}

// Make a sequence of simple translation-only poses.
static void makePoses(std::uint64_t n, std::vector<Pose>& poses)
{
    poses.resize(static_cast<std::size_t>(n));
    for(std::uint64_t i = 0; i < n; ++i)
    {
        // Spiral-ish drift through the box
        float t = float(i) * 0.001f;
        poses[std::size_t(i)]
            = Pose{10.0f + 20.0f * std::cos(t), 10.0f + 20.0f * std::sin(t), 5.0f + 10.0f * std::sin(0.5f * t)};
    }
}

int main(int argc, char** argv)
{
    auto const args = parseArgs(argc, argv);

    std::vector<Atom> ligand, protein;
    makeSynthetic(args.natlig, args.natpro, ligand, protein);

    std::vector<Pose> poses;
    makePoses(args.poses, poses);

    SystemView sys{ligand.data(), ligand.size(), protein.data(), protein.size()};

    std::cout << "miniBUDE CPU physics (synthetic deck)\n"
              << "poses=" << args.poses << " runs=" << args.runs << " natlig=" << args.natlig
              << " natpro=" << args.natpro << "\n";

    std::vector<double> times_ms;
    times_ms.reserve(args.runs);

    // Warm-up
    float volatile sink = 0.0f;
    for(int w = 0; w < 1; ++w)
    {
        for(auto const& p : poses)
            sink += score_pose(sys, p);
    }

    for(int r = 0; r < args.runs; ++r)
    {
        double t = time_ms(
            [&]
            {
                float Eacc = 0.0f;
                for(auto const& p : poses)
                    Eacc += score_pose(sys, p);
                sink += Eacc; // keep it alive
            });
        times_ms.push_back(t);
    }

    // Basic stats over runs: min / avg / max time in ms.
    double mn = std::numeric_limits<double>::infinity();
    double mx = 0.0, sum = 0.0;
    for(double t : times_ms)
    {
        mn = std::min(mn, t);
        mx = std::max(mx, t);
        sum += t;
    }
    double avg = sum / times_ms.size();

    std::cout << std::fixed << std::setprecision(3) << "time_ms: min/avg/max = " << mn << " / " << avg << " / " << mx
              << "\n";

    (void) sink; // silence “unused” warning in case optimizations differ
    return 0;
}
