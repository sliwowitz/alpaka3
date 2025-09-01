// SPDX-License-Identifier: Apache-2.0
// Portions adapted conceptually from UoB-HPC/miniBUDE (Apache-2.0).
// Author: Ivan Andriievskyi
// Work funded by US NAS and ONRG (IMPRESS-U).

#include "physics.hpp"

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
#include <vector>

// ----------------------------
// miniBUDE deck reader (inline)
// ----------------------------

// File names inside the deck directory
static constexpr char const* FILE_LIGAND = "ligand.in";
static constexpr char const* FILE_PROTEIN = "protein.in";
static constexpr char const* FILE_FORCEFIELD = "forcefield.in";
static constexpr char const* FILE_POSES = "poses.in";
static constexpr char const* FILE_REF_ENERGIES = "ref_energies.out";

static inline std::string path_join(std::string const& dir, std::string const& file)
{
    if(dir.empty())
        return file;
    char last = dir.back();
    if(last == '/' || last == '\\')
        return dir + file;
    return dir + "/" + file;
}

// Raw structs as stored by miniBUDE decks (packed to match binary layout)
#pragma pack(push, 1)

struct MBAtom
{
    float x, y, z;
    int32_t type;
};

struct MBFFParams
{
    int32_t hbtype;
    float radius;
    float hphb;
    float elsc;
};

#pragma pack(pop)

template<class T>
static std::vector<T> read_bin_array(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if(!f)
        throw std::runtime_error("open fail: " + path);
    f.seekg(0, std::ios::end);
    std::streamsize len = f.tellg();
    f.seekg(0, std::ios::beg);
    if(len % std::streamsize(sizeof(T)) != 0)
        throw std::runtime_error("size mismatch: " + path);
    std::vector<T> v(static_cast<size_t>(len) / sizeof(T));
    f.read(reinterpret_cast<char*>(v.data()), len);
    if(!f)
        throw std::runtime_error("short read: " + path);
    return v;
}

static std::vector<float> read_bin_floats(std::string const& path)
{
    std::ifstream f(path, std::ios::binary);
    if(!f)
        throw std::runtime_error("open fail: " + path);
    f.seekg(0, std::ios::end);
    std::streamsize len = f.tellg();
    f.seekg(0, std::ios::beg);
    if(len % std::streamsize(sizeof(float)) != 0)
        throw std::runtime_error("size mismatch (floats): " + path);
    std::vector<float> v(static_cast<size_t>(len) / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), len);
    if(!f)
        throw std::runtime_error("short read: " + path);
    return v;
}

// Text fallbacks (for text-formatted decks)
static std::vector<MBAtom> read_text_atoms(std::string const& path)
{
    std::ifstream f(path);
    if(!f)
        throw std::runtime_error("open fail: " + path);
    std::vector<MBAtom> v;
    MBAtom a{};
    while(f >> a.x >> a.y >> a.z >> a.type)
        v.push_back(a);
    return v;
}

static std::vector<MBFFParams> read_text_ff(std::string const& path)
{
    std::ifstream f(path);
    if(!f)
        throw std::runtime_error("open fail: " + path);
    std::vector<MBFFParams> v;
    MBFFParams p{};
    while(f >> p.hbtype >> p.radius >> p.hphb >> p.elsc)
        v.push_back(p);
    return v;
}

static std::vector<float> read_text_floats(std::string const& path)
{
    std::ifstream f(path);
    if(!f)
        return {};
    std::vector<float> xs;
    float t;
    while(f >> t)
        xs.push_back(t);
    return xs;
}

// Auto-detect: try binary, then text
template<class T, class TextReader>
static std::vector<T> read_auto(std::string const& path, TextReader text_reader)
{
    try
    {
        return read_bin_array<T>(path);
    }
    catch(...)
    {
        return text_reader(path);
    }
}

static std::vector<float> read_floats_auto(std::string const& path)
{
    try
    {
        return read_bin_floats(path);
    }
    catch(...)
    {
        return read_text_floats(path);
    }
}

struct MBDeckRaw
{
    std::vector<MBAtom> ligand, protein;
    std::vector<MBFFParams> ff;
    std::array<std::vector<float>, 6> poses6;
    std::vector<float> refEnergies; // may be empty
};

static MBDeckRaw read_mb_deck(std::string const& dir)
{
    MBDeckRaw r;
    auto const pLig = path_join(dir, FILE_LIGAND);
    auto const pPro = path_join(dir, FILE_PROTEIN);
    auto const pFF = path_join(dir, FILE_FORCEFIELD);
    auto const pPos = path_join(dir, FILE_POSES);
    auto const pRef = path_join(dir, FILE_REF_ENERGIES);

    r.ligand = read_auto<MBAtom>(pLig, read_text_atoms);
    r.protein = read_auto<MBAtom>(pPro, read_text_atoms);
    r.ff = read_auto<MBFFParams>(pFF, read_text_ff);

    auto poses_all = read_floats_auto(pPos);
    if(poses_all.size() % 6 != 0)
        throw std::runtime_error("poses length not divisible by 6: " + pPos);
    size_t nposes = poses_all.size() / 6;
    for(int k = 0; k < 6; ++k)
    {
        r.poses6[k].resize(nposes);
        for(size_t i = 0; i < nposes; ++i)
            r.poses6[k][i] = poses_all[k * nposes + i];
    }

    r.refEnergies = read_text_floats(pRef); // optional
    return r;
}

// ----------------------------
// CLI + physics
// ----------------------------

struct Args
{
    std::uint64_t poses = 500000; // number of poses to score
    int runs = 5; // repeats for timing
    std::size_t natlig = 256; // ligand atoms (synthetic mode)
    std::size_t natpro = 4096; // protein atoms (synthetic mode)
    std::string deckDir; // if set => read real deck and print stats
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
        auto getStr = [&](std::string& dst)
        {
            if(i + 1 < argc)
                dst = argv[++i];
        };

        if(s == "--poses")
            getU64(a.poses);
        else if(s == "--runs")
            getI(a.runs);
        else if(s == "--natlig")
            getZ(a.natlig);
        else if(s == "--natpro")
            getZ(a.natpro);
        else if(s == "--deck")
            getStr(a.deckDir);
        else if(s == "-h" || s == "--help")
        {
            std::cout << "miniBUDE (CPU, synthetic deck)\n"
                      << "  --poses  <N>      number of poses (default " << a.poses << ")\n"
                      << "  --runs   <R>      timed repeats (default " << a.runs << ")\n"
                      << "  --natlig <L>      ligand atoms (default " << a.natlig << ")\n"
                      << "  --natpro <P>      protein atoms (default " << a.natpro << ")\n"
                      << "  --deck   <DIR>    read real miniBUDE deck from DIR and print stats\n";
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

    // Deck mode: read real miniBUDE deck and print stats, then exit.
    if(!args.deckDir.empty())
    {
        try
        {
            MBDeckRaw D = read_mb_deck(args.deckDir);
            std::cout << "miniBUDE deck mode\n"
                      << "  ligand atoms : " << D.ligand.size() << "\n"
                      << "  protein atoms: " << D.protein.size() << "\n"
                      << "  ff params    : " << D.ff.size() << "\n"
                      << "  poses        : " << D.poses6[0].size() << "\n"
                      << "  ref energies : " << D.refEnergies.size() << "\n";
            if(!D.poses6[0].empty())
            {
                size_t i = 0;
                std::cout << "  pose[0] = { " << D.poses6[0][i] << ", " << D.poses6[1][i] << ", " << D.poses6[2][i]
                          << ", " << D.poses6[3][i] << ", " << D.poses6[4][i] << ", " << D.poses6[5][i] << " }\n";
            }
            return 0; // I/O-only mode
        }
        catch(std::exception const& e)
        {
            std::cerr << "deck read error: " << e.what() << "\n";
            return 2;
        }
    }

    // Synthetic path
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
