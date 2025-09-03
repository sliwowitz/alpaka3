// SPDX-License-Identifier: Apache-2.0
// Portions adapted conceptually from UoB-HPC/miniBUDE (Apache-2.0).
// Author: Ivan Andriievskyi
// Work funded by US NAS and ONRG (IMPRESS-U).

#include "physics.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
    // common
    std::uint64_t poses = 500000; // synthetic: number of poses
    int runs = 5; // repeats for timing
    std::size_t natlig = 256; // synthetic ligand atoms
    std::size_t natpro = 4096; // synthetic protein atoms
    // deck
    std::string deckDir; // if non-empty -> deck mode
    // verification
    bool verify = false; // compare energies vs ref_energies.out (deck mode)
    double tol_pct = 0.025; // tolerance in %
    std::size_t rows = 8; // how many mismatches to print
    std::string dump_path; // where to dump computed energies
    bool csv = false; // csv output for results
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
                      << "  --tol-pct <pct>     tolerance percent for verification (default " << a.tol_pct << ")\n"
                      << "  --rows <N>          number of mismatches to print (default " << a.rows << ")\n"
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

// -----------------------------------------------
// Synthetic data
// -----------------------------------------------
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

// --------------------
// Deck file structures
// --------------------
#pragma pack(push, 1)

struct DeckAtom
{
    float x, y, z;
    std::int32_t type;
};

struct DeckFF
{
    std::int32_t hbtype;
    float radius;
    float hphb;
    float elsc;
};

#pragma pack(pop)

template<class T>
static std::vector<T> readBinaryVec(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if(!f)
        throw std::runtime_error("Cannot open file: " + path);
    f.seekg(0, std::ios::end);
    std::streampos len = f.tellg();
    if(len % std::streamoff(sizeof(T)) != 0)
        throw std::runtime_error("Bad file size for " + path);
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
        if(line.empty())
            continue;
        v.push_back(std::stof(line));
    }
    return v;
}

// Build poses (Tx,Ty,Tz) from deck DOF arrays.
// NOTE: We currently ignore rotations (Rx,Ry,Rz). Verification will not pass
// until full physics (incl. rotations & all terms) is added.
static void makePosesFromDeck(std::array<std::vector<float>, 6> const& dof, std::vector<Pose>& poses)
{
    std::size_t const n = dof[0].size();
    poses.resize(n);
    for(std::size_t i = 0; i < n; ++i)
    {
        poses[i] = Pose{dof[3][i], dof[4][i], dof[5][i]}; // tx, ty, tz
    }
}

// Convert deck atoms (+ FF table) to our Atom (for the current simplified physics).
// Mapping choice (temporary!):
//   q      <- elsc
//   sigma  <- radius
//   epsilon<- 0.0  (LJ disabled in this loader)
static void deckToPhysics(
    std::vector<DeckAtom> const& ligand_d,
    std::vector<DeckAtom> const& protein_d,
    std::vector<DeckFF> const& ff,
    std::vector<Atom>& ligand,
    std::vector<Atom>& protein)
{
    ligand.resize(ligand_d.size());
    for(std::size_t i = 0; i < ligand_d.size(); ++i)
    {
        auto const& Ld = ligand_d[i];
        auto const& F = ff[static_cast<std::size_t>(Ld.type)];
        ligand[i] = Atom{Ld.x, Ld.y, Ld.z, F.elsc, F.radius, 0.0f};
    }
    protein.resize(protein_d.size());
    for(std::size_t j = 0; j < protein_d.size(); ++j)
    {
        auto const& Pd = protein_d[j];
        auto const& F = ff[static_cast<std::size_t>(Pd.type)];
        protein[j] = Atom{Pd.x, Pd.y, Pd.z, F.elsc, F.radius, 0.0f};
    }
}

// --------------------
// Verification helpers
// --------------------
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
    if(!csv)
    {
        if(!bad.empty())
        {
            std::cerr << "# verification: first " << bad.size() << " mismatches (idx, got, ref, diff_%):\n";
            for(auto idx : bad)
            {
                double const denom = std::max(1.0, std::abs(double(ref[idx])));
                double const d = std::abs(double(got[idx]) - double(ref[idx])) / denom * 100.0;
                std::cerr << "  " << idx << ", " << got[idx] << ", " << ref[idx] << ", " << d << "\n";
            }
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

    std::vector<Atom> ligand, protein;
    std::vector<Pose> poses;

    bool deckMode = !args.deckDir.empty();

    if(!deckMode)
    {
        // Synthetic path
        makeSynthetic(args.natlig, args.natpro, ligand, protein);
        makePoses(args.poses, poses);
        std::cout << "miniBUDE CPU physics (synthetic deck)\n"
                  << "poses=" << args.poses << " runs=" << args.runs << " natlig=" << args.natlig
                  << " natpro=" << args.natpro << "\n";
    }
    else
    {
        // Deck path
        std::string const& D = args.deckDir;
        auto ligand_d = readBinaryVec<DeckAtom>(D + "/ligand.in");
        auto protein_d = readBinaryVec<DeckAtom>(D + "/protein.in");
        auto ff = readBinaryVec<DeckFF>(D + "/forcefield.in");
        std::array<std::vector<float>, 6> dof;
        dof[0] = readBinaryVec<float>(D + "/poses.in");
        // The deck stores poses as 6 contiguous blocks (rx,ry,rz,tx,ty,tz).
        // The loader reads the entire file and splits it evenly into 6 arrays:
        {
            auto const flat = std::move(dof[0]);
            if(flat.size() % 6 != 0)
                throw std::runtime_error("poses.in size not divisible by 6");
            std::size_t const n = flat.size() / 6;
            for(int k = 0; k < 6; ++k)
                dof[k].resize(n);
            for(std::size_t i = 0; i < n; ++i)
            {
                for(int k = 0; k < 6; ++k)
                {
                    dof[k][i] = flat[i + std::size_t(k) * n];
                }
            }
        }
        std::vector<float> refE;
        if(args.verify)
        {
            refE = readRefEnergiesTxt(D + "/ref_energies.out");
        }
        // Map deck fields to the physics parameters used here
        deckToPhysics(ligand_d, protein_d, ff, ligand, protein);
        makePosesFromDeck(dof, poses);

        std::cout << "miniBUDE deck mode\n"
                  << "  ligand atoms : " << ligand_d.size() << "\n"
                  << "  protein atoms: " << protein_d.size() << "\n"
                  << "  ff params    : " << ff.size() << "\n"
                  << "  poses        : " << poses.size() << "\n"
                  << (args.verify ? "  ref energies : " : "") << (args.verify ? std::to_string(refE.size()) : "")
                  << (args.verify ? "\n" : "");
        if(!poses.empty())
        {
            std::cout << "  pose[0] = { " << dof[0][0] << ", " << dof[1][0] << ", " << dof[2][0] << ", "
                      << (int) dof[3][0] << ", " << (int) dof[4][0] << ", " << (int) dof[5][0] << " }\n";
        }
        if(args.verify && refE.size() != poses.size())
        {
            std::cerr << "# WARNING: ref_energies count (" << refE.size() << ") != poses count (" << poses.size()
                      << ")\n";
        }
    }

    // Build a read-only view for scoring
    SystemView sys{ligand.data(), ligand.size(), protein.data(), protein.size()};

    // --- Warm-up (no timing, avoid dead code elimination)
    float volatile sink = 0.0f;
    for(int w = 0; w < 1; ++w)
    {
        for(auto const& p : poses)
            sink += score_pose(sys, p);
    }

    // --- Timed runs (aggregate only for throughput stat)
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

    // --- Basic timing stats
    double mn = std::numeric_limits<double>::infinity();
    double mx = 0.0, sum = 0.0;
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

    // --- Per-pose energies (for verification / dump)
    std::vector<float> energies;
    energies.resize(poses.size());
    for(std::size_t i = 0; i < poses.size(); ++i)
    {
        energies[i] = score_pose(sys, poses[i]);
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

    if(deckMode && args.verify)
    {
        auto const ref = readRefEnergiesTxt(args.deckDir + "/ref_energies.out");
        auto res = verifyEnergies(ref, energies, args.tol_pct, args.rows, args.csv);
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
            if(!res.valid)
            {
                std::cout << "# NOTE: current physics is simplified (no rotations; LJ/Coulomb mapping); "
                             "discrepancy expected until full model is implemented.\n";
            }
        }
    }

    (void) sink;
    return 0;
}
