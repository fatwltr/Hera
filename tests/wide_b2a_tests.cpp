#include "wide_b2a.h"
#include "OT/ot_pack.h"
#include "utils/io_pack.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
using U128 = unsigned __int128;
U128 Raw(hera::WideRingShare x) { return (U128{x.high} << 64) | x.low; }
void Transfer(int fd, void *data, std::size_t size, bool send)
{
    auto *cursor = static_cast<char *>(data);
    while (size) {
        const auto n = send ? write(fd, cursor, size) : read(fd, cursor, size);
        if (n <= 0) throw std::runtime_error("wide B2A test pipe failed");
        cursor += n; size -= n;
    }
}
struct Measurement { double ms{}; std::uint64_t bytes{}; };

// Previous double-candidate wire format, retained solely as an independent
// test/benchmark reference. Randomness here is public test data.
std::vector<hera::WideRingShare> ReferenceGot(
    int party, sci::OTPack &ot, const std::vector<std::uint8_t> &bits, int width)
{
    const auto n = bits.size();
    const int limb_bits = (width + 1) / 2;
    const auto mask = limb_bits == 64 ? UINT64_MAX
        : (std::uint64_t{1} << limb_bits) - 1;
    std::vector<hera::WideRingShare> result(n);
    if (party == sci::ALICE) {
        std::vector<std::uint64_t> messages(4 * n);
        for (std::size_t i = 0; i < n; ++i) {
            result[i] = hera::WideAdd({0x971238abcdefULL + i, 0x7654321ULL + i}, {}, width);
            const auto x0 = Raw(hera::WideSubtract(hera::WideFromLow(bits[i], width), result[i], width));
            const auto x1 = Raw(hera::WideSubtract(hera::WideFromLow(1 - bits[i], width), result[i], width));
            messages[4*i] = std::uint64_t(x0) & mask;
            messages[4*i+1] = std::uint64_t(x1) & mask;
            messages[4*i+2] = std::uint64_t(x0 >> limb_bits) & mask;
            messages[4*i+3] = std::uint64_t(x1 >> limb_bits) & mask;
        }
        ot.iknp_straight->send_batched_got(messages.data(), n, limb_bits, 2);
        ot.iknp_straight->ferret->io->flush();
    } else {
        std::vector<std::uint64_t> received(2*n);
        ot.iknp_straight->recv_batched_got(received.data(), bits.data(), n, limb_bits, 2);
        for (std::size_t i = 0; i < n; ++i)
            result[i] = hera::WideAdd(hera::WideFromLow(received[2*i], width),
                hera::WideShiftLeft(hera::WideFromLow(received[2*i+1], width), limb_bits, width), width);
    }
    return result;
}

void Run(int party, int pipe_fd)
{
    sci::IOPack io(party, 23120, "127.0.0.1");
    sci::OTPack ot(&io, party);
    for (int width : {65, 83, 100, 116, 127}) {
        for (std::size_t n : {std::size_t{1}, std::size_t{7}, std::size_t{257}}) {
            std::vector<std::uint8_t> local(n);
            for (std::size_t i = 0; i < n; ++i)
                local[i] = party == sci::ALICE ? (i / 2) & 1U : i & 1U;
            auto result = hera::conversion_internal::BooleanToWideArithmetic(party, ot, local, width);
            if (party == sci::BOB) Transfer(pipe_fd, result.data(), n*sizeof(result[0]), true);
            else {
                std::vector<hera::WideRingShare> other(n);
                Transfer(pipe_fd, other.data(), n*sizeof(other[0]), false);
                for (std::size_t i = 0; i < n; ++i)
                    if (Raw(hera::WideAdd(result[i], other[i], width)) != (((i/2)&1U) ^ (i&1U)))
                        throw std::runtime_error("wide B2A carry reconstruction failed");
            }
        }
    }
    constexpr std::size_t n = 65536;
    for (bool legacy : {true, false}) {
        double sum_ms = 0; std::uint64_t sum_bytes = 0;
        std::vector<std::uint8_t> local(n);
        for (std::size_t i = 0; i < n; ++i)
            local[i] = party == sci::ALICE ? (i/2)&1U : i&1U;
        for (int round = 0; round < 4; ++round) {
            io.io->sync();
            const auto start = std::chrono::steady_clock::now();
            const auto before_bytes = io.get_comm();
            auto result = legacy ? ReferenceGot(party, ot, local, 100)
                : hera::conversion_internal::BooleanToWideArithmetic(party, ot, local, 100);
            Measurement measurement{std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now()-start).count(), io.get_comm()-before_bytes};
            if (party == sci::BOB) {
                Transfer(pipe_fd, &measurement, sizeof(measurement), true);
                Transfer(pipe_fd, result.data(), n*sizeof(result[0]), true);
            } else {
                Measurement other_measurement;
                Transfer(pipe_fd, &other_measurement, sizeof(other_measurement), false);
                std::vector<hera::WideRingShare> other(n);
                Transfer(pipe_fd, other.data(), n*sizeof(other[0]), false);
                for (std::size_t i = 0; i < n; ++i)
                    if (Raw(hera::WideAdd(result[i], other[i], 100)) != (((i/2)&1U) ^ (i&1U)))
                        throw std::runtime_error("wide B2A batch reconstruction failed");
                if (round) {
                    sum_ms += std::max(measurement.ms, other_measurement.ms);
                    sum_bytes += measurement.bytes + other_measurement.bytes;
                }
            }
        }
        if (party == sci::ALICE)
            std::cout << "WIDE_B2A_BENCH,legacy=" << legacy << ",width=100,elements=" << n
                << ",mean_ms=" << sum_ms/3 << ",both_mean_bytes=" << sum_bytes/3 << '\n';
    }
}
} // namespace

int main()
{
    int pipe_fd[2]; if (pipe(pipe_fd)) return 1;
    const auto child = fork(); if (child < 0) return 1;
    if (child == 0) {
        close(pipe_fd[0]);
        try { Run(sci::BOB, pipe_fd[1]); _exit(0); }
        catch (const std::exception &e) { std::cerr << e.what() << '\n'; _exit(2); }
    }
    close(pipe_fd[1]);
    try {
        Run(sci::ALICE, pipe_fd[0]);
        int status; waitpid(child, &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status)) return 2;
        std::cout << "HERA_WIDE_B2A_STATUS=PASS\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 2; }
}
