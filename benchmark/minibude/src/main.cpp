// SPDX-License-Identifier: Apache-2.0
// Portions adapted conceptually from UoB-HPC/miniBUDE (Apache-2.0).
// Author: Ivan Andriievskyi
// Work funded by US NAS and ONRG (IMPRESS-U).

#include "physics.hpp" // synthetic LJ+Coulomb path
#include "physics_minibude.hpp" // full miniBUDE CPU serial physics

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// -------------------------------
// CLI
// -------------------------------
struct Args
{
    // synthetic
    std::uint64_t poses = 500000;
    int runs = 5;
    std::size_t natlig = 256;
    std::size_t natpro = 4096;
    // deck
    std::string deckDir;
    // verification
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
        auto next_u64 = [&](std::uint64_t& dst)
        {
            if(i + 1 < argc)
                dst = std::stoull(argv[++i]);
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

        if(s == "--poses")
            next_u64(a.poses);
        else if(s == "--runs")
            next_i(a.runs);
        else if(s == "--natlig")
            next_z(a.natlig);
        else if(s == "--natpro")
            next_z(a.natpro);
        else if(s == "--deck")
            next_str(a.deckDir);
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
            std::cout << "miniBUDE (CPU)\n\n"
                      << "SYNTHETIC MODE:\n"
                      << "  --poses <N>         number of poses (default " << a.poses << ")\n"
                      << "  --runs  <R>         timed repeats (default " << a.runs << ")\n"
                      << "  --natlig <L>        ligand atoms (default " << a.natlig << ")\n"
                      << "  --natpro <P>        protein atoms (default " << a.natpro << ")\n\n"
                      << "DECK MODE:\n"
                      << "  --deck <DIR>        directory with bm1/{ligand,protein,forcefield,poses,ref_energies}\n"
                      << "  --verify            compare energies against ref_energies.out\n"
                      << "  --tol-pct <pct>     tolerance percent (default " << a.tol_pct << ")\n"
                      << "  --rows <N>          how many mismatches to print (default " << a.rows << ")\n"
                      << "  --dump-energies P   write computed energies to file P\n"
                      << "  --csv               CSV summary output\n";
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

// ---------------- Synthetic data ---------------
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
    std::uniform_real_distribution<float> sdist(2.5f, 4.5f);
    std::uniform_real_distribution<float> edist(0.01f, 0.20f);

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

static void makePoses(std::uint64_t n, std::vector<Pose>& poses)
{
    poses.resize(static_cast<std::size_t>(n));
    for(std::uint64_t i = 0; i < n; ++i)
    {
        float t = float(i) * 0.001f;
        poses[std::size_t(i)]
            = Pose{10.0f + 20.0f * std::cos(t), 10.0f + 20.0f * std::sin(t), 5.0f + 10.0f * std::sin(0.5f * t)};
    }
}

// ---------------- Deck I/O helpers ------------------------------------------
template<class T>
static std::vector<T> readBinaryVec(std::string const& path)
{
    std::ifstream f(path, std::ios::binary);
    if(!f)
        throw std::runtime_error("Cannot open file: " + path);
    f.seekg(0, std::ios::end);
    auto len = f.tellg();
    if(len % std::streamoff(sizeof(T)) != 0)
        throw std::runtime_error("Bad file size: " + path);
    std::vector<T> v(static_cast<std::size_t>(len) / sizeof(T));
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(len));
    return v;
}

static std::vector<float> readRefEnergiesTxt(std::string const& path)
{
    std::ifstream in(path);
    if(!in)
        throw std::runtime_error("Cannot open ref energies: " + path);
    std::vector<float> v;
    std::string line;
    while(std::getline(in, line))
    {
        if(!line.empty())
            v.push_back(std::stof(line));
    }
    return v;
}

static void splitPoses6(std::vector<float> const& flat, std::array<std::vector<float>, 6>& dof)
{
    if(flat.size() % 6 != 0)
        throw std::runtime_error("poses.in size not divisible by 6");
    std::size_t const n = flat.size() / 6;
    for(int k = 0; k < 6; ++k)
        dof[k].resize(n);
    for(std::size_t i = 0; i < n; ++i)
        for(int k = 0; k < 6; ++k)
            dof[k][i] = flat[i + std::size_t(k) * n];
}

// ---------------- Verification ----------------------------------------------
struct VerifyResult
{
    bool valid;
    double max_diff_pct;
    std::vector<std::size_t> bad_idx;
};

static VerifyResult verifyEnergies(
    std::vector<float> const& ref,
    std::vector<float> const& got,
    double tol_pct,
    std::size_t rows,
    bool csv)
{
    std::size_t const n = std::min(ref.size(), got.size());
    double maxd = 0.0;
    std::vector<std::size_t> bad;
    for(std::size_t i = 0; i < n; ++i)
    {
        double const denom = std::max(1.0, std::abs(double(ref[i])));
        double const d = std::abs(double(got[i]) - double(ref[i])) / denom * 100.0;
        if(d > maxd)
            maxd = d;
        if(d > tol_pct && bad.size() < rows)
            bad.push_back(i);
    }
    if(!csv && !bad.empty())
    {
        std::cerr << "# verification: first " << bad.size() << " mismatches (idx, got, ref, diff_%):\n";
        for(auto idx : bad)
        {
            double const denom = std::max(1.0, std::abs(double(ref[idx])));
            double const d = std::abs(double(got[idx]) - double(ref[idx])) * 100.0 / denom;
            std::cerr << "  " << idx << ", " << got[idx] << ", " << ref[idx] << ", " << d << "\n";
        }
    }
    return VerifyResult{maxd <= tol_pct, maxd, std::move(bad)};
}

// -------------
// Program entry
// -------------
int main(int argc, char** argv)
{
    auto const args = parseArgs(argc, argv);

    bool const deckMode = !args.deckDir.empty();

    // Synthetic path ( ------------------------------------
    if(!deckMode)
    {
        std::vector<Atom> ligand, protein;
        std::vector<Pose> poses;
        makeSynthetic(args.natlig, args.natpro, ligand, protein);
        makePoses(args.poses, poses);

        SystemView sys{ligand.data(), ligand.size(), protein.data(), protein.size()};

        std::cout << "miniBUDE CPU physics (synthetic deck)\n"
                  << "poses=" << args.poses << " runs=" << args.runs << " natlig=" << args.natlig
                  << " natpro=" << args.natpro << "\n";

        float volatile sink = 0.0f;
        // warmup
        for(int w = 0; w < 1; ++w)
            for(auto const& p : poses)
                sink += score_pose(sys, p);

        std::vector<double> times_ms;
        times_ms.reserve(args.runs);
        for(int r = 0; r < args.runs; ++r)
        {
            double t = time_ms(
                [&]
                {
                    float Eacc = 0.0f;
                    for(auto const& p : poses)
                        Eacc += score_pose(sys, p);
                    sink += Eacc;
                });
            times_ms.push_back(t);
        }

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

        (void) sink;
        return 0;
    }

    // Deck path (full miniBUDE physics) --------------------------------------
    std::string const& D = args.deckDir;

    // read deck
    auto const ligand_d = readBinaryVec<DeckAtom>(D + "/ligand.in");
    auto const protein_d = readBinaryVec<DeckAtom>(D + "/protein.in");
    auto const ff = readBinaryVec<FFParams>(D + "/forcefield.in");

    std::array<std::vector<float>, 6> dof;
    {
        auto const flat = readBinaryVec<float>(D + "/poses.in");
        splitPoses6(flat, dof); // dof[0..5] = {rx, ry, rz, tx, ty, tz}
    }

    std::vector<float> refE;
    if(args.verify)
    {
        refE = readRefEnergiesTxt(D + "/ref_energies.out");
    }

    std::cout << "miniBUDE deck mode\n"
              << "  ligand atoms : " << ligand_d.size() << "\n"
              << "  protein atoms: " << protein_d.size() << "\n"
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

    // warmup
    float volatile sink = 0.0f;
    for(int w = 0; w < 1; ++w)
    {
        for(std::size_t i = 0; i < N; ++i)
        {
            sink += score_pose_minibude_serial(
                ligand_d.data(),
                ligand_d.size(),
                protein_d.data(),
                protein_d.size(),
                ff.data(),
                dof[0][i],
                dof[1][i],
                dof[2][i],
                dof[3][i],
                dof[4][i],
                dof[5][i]);
        }
    }

    // timed runs
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
                        ligand_d.data(),
                        ligand_d.size(),
                        protein_d.data(),
                        protein_d.size(),
                        ff.data(),
                        dof[0][i],
                        dof[1][i],
                        dof[2][i],
                        dof[3][i],
                        dof[4][i],
                        dof[5][i]);
                    Eacc += e;
                    if(r == args.runs - 1)
                        energies[i] = e; // keep last run as result
                }
                sink += Eacc;
            });
        times_ms.push_back(t);
    }

    // timing summary
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

    // dump energies if requested
    if(!args.dump_path.empty())
    {
        std::ofstream out(args.dump_path, std::ios::out | std::ios::trunc);
        for(float e : energies)
            out << std::setprecision(7) << e << "\n";
        std::cout
            << (out ? "# energies dumped to " + args.dump_path + "\n"
                    : "# ERROR: failed to write " + args.dump_path + "\n");
    }

    // verify if requested
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
