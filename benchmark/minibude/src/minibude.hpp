// SPDX-License-Identifier: Apache-2.0
// Author: Ivan Andriievskyi
// Work funded by US NAS and ONRG (IMPRESS-U).

#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ===== Types and constants shared with physics =====
#pragma pack(push, 1)

struct DeckAtom
{
    float x, y, z;
    std::int32_t type;
};

struct FFParams
{
    std::int32_t hbtype;
    float radius;
    float hphb;
    float elsc;
};

#pragma pack(pop)

inline constexpr float ZERO = 0.0f;
inline constexpr float QUARTER = 0.25f;
inline constexpr float HALF = 0.5f;
inline constexpr float ONE = 1.0f;
inline constexpr float TWO = 2.0f;
inline constexpr float FOUR = 4.0f;
inline constexpr float CNSTNT = 45.0f;

inline constexpr int HBTYPE_F = 70;
inline constexpr int HBTYPE_E = 69;
inline constexpr float HARDNESS = 38.0f;
inline constexpr float NPNPDIST = 5.5f;
inline constexpr float NPPDIST = 1.0f;

inline constexpr float FloatMax = std::numeric_limits<float>::max();

// ===== Deck I/O =====
template<class T>
inline std::vector<T> readBinaryVec(std::string const& path)
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

inline std::vector<float> readRefEnergiesTxt(std::string const& path)
{
    std::ifstream in(path);
    if(!in)
        throw std::runtime_error("Cannot open ref energies: " + path);
    std::vector<float> v;
    std::string line;
    while(std::getline(in, line))
        if(!line.empty())
            v.push_back(std::stof(line));
    return v;
}

inline void splitPoses6(std::vector<float> const& flat, std::array<std::vector<float>, 6>& dof)
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

// ===== Verification =====
struct VerifyResult
{
    bool valid;
    double max_diff_pct;
    std::vector<std::size_t> bad_idx;
};

inline VerifyResult verifyEnergies(
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
