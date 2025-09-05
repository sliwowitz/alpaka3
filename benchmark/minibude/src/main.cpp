// SPDX-License-Identifier: Apache-2.0
// Portions adapted conceptually from UoB-HPC/miniBUDE (Apache-2.0).
// Author: Ivan Andriievskyi
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

    // Warmup
    float volatile sink = 0.0f;
    for(int w = 0; w < 1; ++w)
    {
        for(std::size_t i = 0; i < N; ++i)
        {
            float e = score_pose_minibude_serial(
                ligand.data(),
                ligand.size(),
                protein.data(),
                protein.size(),
                ff.data(),
                dof[0][i],
                dof[1][i],
                dof[2][i],
                dof[3][i],
                dof[4][i],
                dof[5][i]);
            sink = sink + e;
        }
    }

    // Timed runs
    std::vector<double> times_ms;
    times_ms.reserve(args.runs);
    std::vector<float> energies(N, 0.0f);
    for(int r = 0; r < args.runs; ++r)
    {
        double t = time_ms(
            [&]
            {
                float Eacc = 0.0f;
                for(std::size_t i = 0; i < N; ++i)
                {
                    float e = score_pose_minibude_serial(
                        ligand.data(),
                        ligand.size(),
                        protein.data(),
                        protein.size(),
                        ff.data(),
                        dof[0][i],
                        dof[1][i],
                        dof[2][i],
                        dof[3][i],
                        dof[4][i],
                        dof[5][i]);
                    Eacc += e;
                    if(r == args.runs - 1)
                        energies[i] = e;
                }
                sink = sink + Eacc;
            });
        times_ms.push_back(t);
    }

    // Timing summary
    double mn = std::numeric_limits<double>::infinity(), mx = 0.0, sum = 0.0;
    for(double t : times_ms)
    {
        mn = std::min(mn, t);
        mx = std::max(mx, t);
        sum += t;
    }
    double avg = sum / times_ms.size();

    if(args.csv)
    {
        std::cout << std::fixed << std::setprecision(3) << "min_ms,avg_ms,max_ms\n"
                  << mn << "," << avg << "," << mx << "\n";
    }
    else
    {
        std::cout << std::fixed << std::setprecision(3) << "time_ms: min/avg/max = " << mn << " / " << avg << " / "
                  << mx << "\n";
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
