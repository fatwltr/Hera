#include "bounded_carry.h"
#include "OT/ot_pack.h"
#include "utils/io_pack.h"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using U128 = unsigned __int128;
U128 Raw(hera::WideRingShare x) { return (U128{x.high} << 64) | x.low; }
void Check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
void Transfer(int fd, void *buffer, std::size_t size, bool sending)
{
    auto *p = static_cast<char *>(buffer);
    while (size) {
        const auto n = sending ? write(fd, p, size) : read(fd, p, size);
        Check(n > 0, "carry oracle pipe failed");
        p += n; size -= n;
    }
}
void Verify(int party, int fd, const std::vector<hera::WideRingShare> &result,
            int bits, const std::vector<U128> &expected)
{
    auto peer = result;
    if (party == sci::BOB) Transfer(fd, peer.data(), peer.size()*sizeof(peer[0]), true);
    else {
        Transfer(fd, peer.data(), peer.size()*sizeof(peer[0]), false);
        const auto mask = (U128{1} << bits) - 1;
        for (std::size_t i = 0; i < result.size(); ++i)
            Check(Raw(hera::WideAdd(result[i], peer[i], bits)) == (expected[i] & mask),
                  "arithmetic OR or prime-to-ring differs from integer oracle");
    }
}
std::uint64_t Next(std::uint64_t &state)
{
    state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state;
}
std::vector<std::uint64_t> FieldShares(int party, std::uint64_t q, std::size_t n,
                                     std::vector<U128> &expected)
{
    std::uint64_t state = 0x4387abcd01923ULL;
    std::vector<std::uint64_t> shares(n);
    expected.resize(n);
    const auto c = static_cast<std::int64_t>(q / 4);
    for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t values[] = {0, 1, -1, c-1, -c+1};
        const auto value = values[i % 5];
        auto a = Next(state) % q;
        if (i % 11 == 0) a = 0;
        if (i % 11 == 1) a = q-1;
        if (i % 11 == 2) a = q/2;
        const auto clear = value < 0 ? q-static_cast<std::uint64_t>(-value) : value;
        shares[i] = party == sci::ALICE ? a : (clear + q - a) % q;
        expected[i] = static_cast<U128>(value);
    }
    return shares;
}

struct Measurement { double ms{}; std::uint64_t bytes{}; };
void Benchmark(int party, sci::IOPack &io, int fd)
{
    constexpr std::size_t n = 16384;
    constexpr std::uint64_t q = 1152921504606830593ULL;
    std::vector<U128> expected;
    auto local = FieldShares(party, q, n, expected);
    for (bool bounded : {true}) {
        // Independent OT state for each algorithm, excluding initialization
        // and the warmup. No carry pads are shared between experiments.
        const auto setup_started = std::chrono::steady_clock::now();
        sci::OTPack ot(&io, party);
        const auto setup_ms = std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-setup_started).count();
        double total_ms = 0;
        std::uint64_t total_bytes = 0;
        // A short bounded run fits entirely in Ferret's startup buffer.
        // Consume exactly three buffers, ending at the initial OT position,
        // so prepaid material is not a hidden subsidy to the measured work.
        const auto measured_elements = bounded
            ? 3 * static_cast<std::size_t>(ot.iknp_straight->ferret->ot_limit) : 16 * n;
        const int measured_runs = static_cast<int>((measured_elements + n-1) / n);
        int measured_refills = 0;
        std::int64_t steady_start_used = 0;
        for (int run = 0; run <= measured_runs; ++run) {
            if (bounded && run == measured_runs)
                local.resize(measured_elements - (measured_runs-1) * n);
            io.io->sync();
            const auto before = io.get_comm();
            const auto used_before = ot.iknp_straight->ferret->ot_used;
            const auto start = std::chrono::steady_clock::now();
            std::vector<hera::WideRingShare> result;
            if (bounded) result = hera::conversion_internal::BoundedCenteredPrimeToRing(party, *ot.iknp_straight, local, q, 100);
            Measurement m{std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count(), io.get_comm()-before};
            if (!run) steady_start_used = ot.iknp_straight->ferret->ot_used;
            if (run && ot.iknp_straight->ferret->ot_used < used_before) ++measured_refills;
            if (party == sci::BOB) Transfer(fd, &m, sizeof(m), true);
            else {
                Measurement peer;
                Transfer(fd, &peer, sizeof(peer), false);
                if (run) { total_ms += std::max(m.ms,peer.ms); total_bytes += m.bytes+peer.bytes; }
            }
            Verify(party, fd, result, 100, expected);
        }
        if (party == sci::ALICE)
            std::cout << "CARRY_BENCH,bounded=" << bounded << ",elements=" << n
                      << ",measured_runs=" << measured_runs
                      << ",measured_elements=" << measured_elements
                      << ",straight_refills=" << measured_refills
                      << ",setup_ms=" << setup_ms
                      << ",mean_ms=" << total_ms*n/measured_elements
                      << ",both_mean_bytes=" << total_bytes*n/measured_elements << '\n';
        if (bounded) Check(measured_refills >= 3, "bounded benchmark did not include amortized OT refills");
        if (bounded) Check(ot.iknp_straight->ferret->ot_used == steady_start_used,
            "bounded benchmark changed the amount of prepaid OT material");
    }
}
void Run(int party, int fd, bool benchmark)
{
    sci::IOPack io(party, 23200, "127.0.0.1");
    {
        sci::OTPack ot(&io, party);
        for (int bits : {1, 2, 40, 63, 64, 65, 83, 100, 116, 127}) {
            for (std::size_t n : {std::size_t{0},std::size_t{1},std::size_t{7},std::size_t{257}}) {
                for (auto weight : {std::uint64_t{1},UINT64_MAX,std::uint64_t{1152921504606830593ULL}}) {
                    std::vector<std::uint8_t> local(n);
                    std::vector<U128> expected(n);
                    for (std::size_t i = 0; i < n; ++i) {
                        local[i] = party == sci::ALICE ? (i/2)&1U : i&1U;
                        expected[i] = U128{weight} * (((i/2)&1U) | (i&1U));
                    }
                    auto result = hera::conversion_internal::LocalBitOrToArithmetic(party,*ot.iknp_straight,local,bits,weight);
                    Verify(party,fd,result,bits,expected);
                }
            }
        }
        for (auto q : {std::uint64_t{13},std::uint64_t{97},std::uint64_t{1152921504606830593ULL}}) {
            for (int bits : {40, 64, 65, 100, 127}) {
                std::vector<U128> expected;
                auto local = FieldShares(party,q,257,expected);
                const auto original = local;
                auto result = hera::conversion_internal::BoundedCenteredPrimeToRing(party,*ot.iknp_straight,local,q,bits);
                Check(local == original, "prime-to-ring changed its inputs");
                Verify(party,fd,result,bits,expected);
            }
        }
    }
    if (benchmark) Benchmark(party,io,fd);
}
}

int main(int argc, char **argv)
{
    pid_t child = -1;
    try {
        const bool benchmark = argc == 2 && std::string(argv[1]) == "--benchmark";
        int fds[2]; Check(pipe(fds) == 0, "carry test pipe failed");
        child = fork(); Check(child >= 0, "carry test fork failed");
        if (child == 0) {
            close(fds[0]);
            try { Run(sci::BOB,fds[1],benchmark); _exit(0); }
            catch (const std::exception &e) { std::cerr << e.what() << '\n'; _exit(2); }
        }
        close(fds[1]); Run(sci::ALICE,fds[0],benchmark);
        int status{}; waitpid(child,&status,0); child = -1;
        Check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "carry test peer failed");
        std::cout << "HERA_CONVERSION_CARRY_STATUS=PASS\n";
    } catch (const std::exception &e) {
        if (child > 0) { kill(child,SIGTERM); waitpid(child,nullptr,0); }
        std::cerr << e.what() << '\n'; return 1;
    }
}
