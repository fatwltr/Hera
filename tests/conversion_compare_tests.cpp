#include "Millionaire/millionaire.h"
#include "Millionaire/millionaire_with_equality.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <csignal>
#include <sys/wait.h>
#include <unistd.h>

namespace {
void Check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

void Transfer(int fd, void *buffer, std::size_t size, bool sending) {
    auto *position = static_cast<char *>(buffer);
    while (size) {
        const auto n = sending ? write(fd, position, size) : read(fd, position, size);
        Check(n > 0, "comparison oracle pipe failed");
        position += n;
        size -= static_cast<std::size_t>(n);
    }
}

std::uint64_t Next(std::uint64_t &state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

void Run(bool alice, int fd) {
    const auto party = alice ? sci::ALICE : sci::BOB;
    sci::IOPack io(party, 17970, "127.0.0.1");
    sci::OTPack ot(&io, party);
    MillionaireProtocol comparison(party, &io, &ot);
    MillionaireWithEquality equality(party, &io, &ot);
    std::size_t checked = 0, sequence = 0;
    const std::size_t sizes[] = {1, 7, 8, 9, 33, 257};
    for (int bits = 1; bits <= 63; ++bits) {
        for (int radix : {3, 4, 5}) for (bool greater : {false, true}) {
            const auto size = sizes[sequence++ % 6];
            const auto mask = (std::uint64_t{1} << bits) - 1;
            const auto midpoint = std::uint64_t{1} << (bits - 1);
            std::uint64_t state = 0x7165abcd891234ULL;
            std::vector<std::uint64_t> left(size), right(size);
            for (std::size_t i = 0; i < size; ++i) {
                left[i] = Next(state) & mask;
                right[i] = Next(state) & mask;
                switch (i % 8) {
                    case 0: right[i] = left[i]; break;
                    case 1: left[i] = 0; right[i] = mask; break;
                    case 2: left[i] = mask; right[i] = 0; break;
                    case 3: left[i] = midpoint; right[i] = midpoint - 1; break;
                    case 4: left[i] = midpoint - 1; right[i] = midpoint; break;
                    case 5: left[i] = mask; right[i] = mask - 1; break;
                    case 6: left[i] = right[i] = 0; break;
                    default: break;
                }
            }
            auto local = alice ? left : right;
            const auto original = local;
            std::vector<std::uint8_t> ordinary(size), correlated(size), eq_cmp(size), eq_result(size);
            comparison.compare(ordinary.data(), local.data(), static_cast<int>(size),
                bits, greater, false, radix);
            comparison.compare_correlated(correlated.data(), local.data(), static_cast<int>(size),
                bits, greater, false, radix);
            Check(local == original, "comparison changed its inputs");
            if (bits > 4 * radix) {
                // _8KKOT reuses the first triple operand only for the shared
                // EQ input. Its two local masked openings must coincide;
                // pairing that mask with different inputs would leak XOR.
                const auto stride = ((size + 7) / 8);
                const auto &openings = hera_sci_detail::comparison_scratch().ei;
                const auto start = comparison.num_triples_std * stride;
                for (int pair = 0; pair < comparison.num_triples_corr / 2; ++pair) {
                    const auto begin = start + 2 * static_cast<std::size_t>(pair) * stride;
                    Check(begin + 2 * stride <= openings.size(), "correlated audit exceeds workspace");
                    Check(std::equal(openings.begin() + begin, openings.begin() + begin + stride,
                        openings.begin() + begin + stride), "correlated masks expose distinct inputs");
                }
            }
            equality.compare_with_eq(eq_cmp.data(), eq_result.data(), local.data(),
                static_cast<int>(size), bits, greater, radix);
            Check(local == original, "comparison/equality changed its inputs");
            if (!alice) {
                Transfer(fd, ordinary.data(), size, true);
                Transfer(fd, correlated.data(), size, true);
                Transfer(fd, eq_cmp.data(), size, true);
                Transfer(fd, eq_result.data(), size, true);
            } else {
                std::vector<std::uint8_t> peer_ordinary(size), peer_correlated(size),
                    peer_eq_cmp(size), peer_eq_result(size);
                Transfer(fd, peer_ordinary.data(), size, false);
                Transfer(fd, peer_correlated.data(), size, false);
                Transfer(fd, peer_eq_cmp.data(), size, false);
                Transfer(fd, peer_eq_result.data(), size, false);
                for (std::size_t i = 0; i < size; ++i) {
                    const auto truth = greater ? left[i] > right[i] : left[i] < right[i];
                    Check((ordinary[i] ^ peer_ordinary[i]) == truth,
                        "ordinary comparison differs from plaintext oracle");
                    Check((correlated[i] ^ peer_correlated[i]) == truth,
                        "correlated comparison differs from plaintext oracle");
                    Check((eq_cmp[i] ^ peer_eq_cmp[i]) == truth,
                        "comparison/equality differs from plaintext comparison oracle");
                    Check((eq_result[i] ^ peer_eq_result[i]) == (left[i] == right[i]),
                        "comparison/equality differs from plaintext equality oracle");
                    ++checked;
                }
            }
        }
    }
    if (alice) std::cout << "COMPARISON_ORACLE_ELEMENTS=" << checked << '\n';
}
} // namespace

int main() {
    pid_t child = -1;
    try {
        int pipe_fd[2];
        Check(pipe(pipe_fd) == 0, "comparison pipe creation failed");
        child = fork();
        Check(child >= 0, "comparison fork failed");
        if (child == 0) {
            close(pipe_fd[0]);
            try { Run(false, pipe_fd[1]); _exit(0); }
            catch (const std::exception &error) { std::cerr << error.what() << '\n'; _exit(2); }
        }
        close(pipe_fd[1]);
        Run(true, pipe_fd[0]);
        int status{};
        Check(waitpid(child, &status, 0) == child, "comparison wait failed");
        child = -1;
        Check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "comparison peer failed");
        std::cout << "HERA_CONVERSION_COMPARE_STATUS=PASS\n";
    } catch (const std::exception &error) {
        if (child > 0) { kill(child, SIGTERM); waitpid(child, nullptr, 0); }
        std::cerr << error.what() << '\n';
        return 1;
    }
}
