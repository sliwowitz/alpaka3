#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct Args
{
    std::uint64_t poses = 1'000'000; // how many “poses” we (pretend to) process
    int runs = 5; // how many timed runs
};

static Args parseArgs(int argc, char** argv)
{
    Args a;
    for(int i = 1; i < argc; ++i)
    {
        std::string s(argv[i]);
        auto getVal = [&](std::uint64_t& dst)
        {
            if(i + 1 < argc)
                dst = std::stoull(argv[++i]);
        };
        auto getInt = [&](int& dst)
        {
            if(i + 1 < argc)
                dst = std::stoi(argv[++i]);
        };
        if(s == "--poses")
            getVal(a.poses);
        else if(s == "--runs")
            getInt(a.runs);
        else if(s == "-h" || s == "--help")
        {
            std::cout << "miniBUDE (skeleton)\n"
                      << "  --poses <N>   number of poses (default " << a.poses << ")\n"
                      << "  --runs  <R>   number of timed runs (default " << a.runs << ")\n";
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

int main(int argc, char** argv)
{
    auto const args = parseArgs(argc, argv);
    std::cout << "miniBUDE skeleton — no physics yet\n"
              << "poses=" << args.poses << " runs=" << args.runs << "\n";

    // Minimal “work”: allocate a scores vector. Later we will plug the miniBUDE kernel here.
    std::vector<double> scores(args.poses, 1.0);

    // Warmup (not timed): avoid cold-start artifacts.
    std::fill(scores.begin(), scores.end(), 0.0);

    std::vector<double> times_ms;
    times_ms.reserve(args.runs);

    // Prevent the compiler from discarding the loop’s work.
    double volatile sink = 0.0;

    for(int r = 0; r < args.runs; ++r)
    {
        double t = time_ms(
            [&]
            {
                std::fill(scores.begin(), scores.end(), 0.0);
                // Use a non-volatile accumulator and a simple assignment to volatile at the end
                // to avoid the “compound assignment to volatile is deprecated” warning.
                double acc = 0.0;
                for(double v : scores)
                    acc += v;
                sink = acc;
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
    double const avg = sum / times_ms.size();

    std::cout << std::fixed << std::setprecision(3) << "time_ms: min/avg/max = " << mn << " / " << avg << " / " << mx
              << "\n";

    (void) sink; // silence “unused” warning in case optimizations differ
    return 0;
}
