#include "OT/ot_pack.h"
#include "FloatingPoint/fixed-point.h"
#include "utils/io_pack.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <sys/wait.h>
#include <utility>
#include <unistd.h>
#include <vector>

namespace {

constexpr int kPort = 20120;
constexpr int kLength = 4;
constexpr int kChoices = 16;
constexpr int kRounds = 2;

struct Packet {
    int ok{};
    std::uint8_t comparison{};
    std::uint64_t selected{};
    std::uint64_t maximum{};
    std::uint64_t product{};
    std::uint64_t straight_receive{};
    std::uint64_t reverse_send{};
    std::uint8_t values[kRounds][kLength]{};
};

std::uint8_t Message(int round, int index, int choice)
{
    return static_cast<std::uint8_t>(17 * round + 19 * index + choice);
}

} // namespace

int main()
{
    try {
        int result_pipe[2];
        if (pipe(result_pipe) != 0)
            throw std::runtime_error("cannot create VOLE OT result pipe");
        const pid_t child = fork();
        if (child < 0) throw std::runtime_error("cannot fork VOLE OT client");
        if (child == 0) {
            close(result_pipe[0]);
            Packet packet;
            try {
                sci::IOPack io(sci::BOB, kPort, "127.0.0.1");
                sci::OTPack ot(&io, sci::BOB);
                FixOp fix(sci::BOB, &io, &ot);
                constexpr std::uint64_t ring_mask =
                    (std::uint64_t{1} << 43) - 1;
                FixArray left(sci::BOB, 1, true, 43, 13);
                FixArray right(sci::BOB, 1, true, 43, 13);
                left.data[0] = ((-static_cast<std::int64_t>(
                                    std::uint64_t{1} << 13)) - 0x12407) &
                               ring_mask;
                right.data[0] = ((std::uint64_t{1} << 13) - 0x12468) &
                                ring_mask;
                FixArray singleton(sci::BOB, 1, true, 43, 13);
                singleton.data[0] = right.data[0];
                std::vector<FixArray> singleton_rows;
                singleton_rows.push_back(std::move(singleton));
                (void)fix.max(singleton_rows);
                FixArray row(sci::BOB, 2, true, 43, 13);
                row.data[0] = left.data[0];
                row.data[1] = right.data[0];
                std::vector<FixArray> rows;
                rows.push_back(std::move(row));
                auto maximum = fix.max(rows);
                packet.maximum = fix.output(sci::PUBLIC, maximum).data[0];
                auto greater = fix.GT(left, right);
                auto revealed_greater = fix.bool_op->output(
                    sci::PUBLIC, greater);
                auto selected = fix.if_else(greater, left, right);
                auto revealed_selected = fix.output(sci::PUBLIC, selected);
                packet.comparison = revealed_greater.data[0];
                packet.selected = revealed_selected.data[0];
                constexpr std::uint64_t short_mask =
                    (std::uint64_t{1} << 19) - 1;
                FixArray factor_a(sci::BOB, 1, false, 19, 13);
                FixArray factor_b(sci::BOB, 1, false, 19, 13);
                factor_a.data[0] = ((std::uint64_t{1} << 13) - 0x12345) &
                                   short_mask;
                factor_b.data[0] = ((std::uint64_t{1} << 13) - 0x23456) &
                                   short_mask;
                packet.product = fix.output(
                    sci::PUBLIC, fix.mul(factor_a, factor_b, 38)).data[0];
                bool straight_choice = true;
                ot.iknp_straight->recv_cot(
                    &packet.straight_receive, &straight_choice, 1, 43);
                const std::uint64_t reverse_correlation = 11;
                ot.iknp_reversed->send_cot(
                    &packet.reverse_send, &reverse_correlation, 1, 43);
                const std::array<std::uint8_t, kLength> choices{0, 3, 9, 15};
                for (int round = 0; round < kRounds; ++round)
                    ot.kkot[3]->recv(packet.values[round], choices.data(),
                                    kLength, 8);
                packet.ok = 1;
            } catch (...) {
                packet.ok = 0;
            }
            const auto written = write(result_pipe[1], &packet, sizeof(packet));
            close(result_pipe[1]);
            _exit(written == static_cast<ssize_t>(sizeof(packet)) && packet.ok
                      ? 0 : 2);
        }

        close(result_pipe[1]);
        sci::IOPack io(sci::ALICE, kPort, {});
        sci::OTPack ot(&io, sci::ALICE);
        FixOp fix(sci::ALICE, &io, &ot);
        FixArray left(sci::ALICE, 1, true, 43, 13);
        FixArray right(sci::ALICE, 1, true, 43, 13);
        left.data[0] = 0x12407;
        right.data[0] = 0x12468;
        FixArray singleton(sci::ALICE, 1, true, 43, 13);
        singleton.data[0] = right.data[0];
        std::vector<FixArray> singleton_rows;
        singleton_rows.push_back(std::move(singleton));
        (void)fix.max(singleton_rows);
        FixArray row(sci::ALICE, 2, true, 43, 13);
        row.data[0] = left.data[0];
        row.data[1] = right.data[0];
        std::vector<FixArray> rows;
        rows.push_back(std::move(row));
        auto maximum = fix.max(rows);
        auto revealed_maximum = fix.output(sci::PUBLIC, maximum);
        auto greater = fix.GT(left, right);
        auto revealed_greater = fix.bool_op->output(sci::PUBLIC, greater);
        auto selected = fix.if_else(greater, left, right);
        auto revealed_selected = fix.output(sci::PUBLIC, selected);
        FixArray factor_a(sci::ALICE, 1, false, 19, 13);
        FixArray factor_b(sci::ALICE, 1, false, 19, 13);
        factor_a.data[0] = 0x12345;
        factor_b.data[0] = 0x23456;
        auto revealed_product = fix.output(
            sci::PUBLIC, fix.mul(factor_a, factor_b, 38));
        const std::uint64_t straight_correlation = 7;
        std::uint64_t straight_send = 0;
        ot.iknp_straight->send_cot(
            &straight_send, &straight_correlation, 1, 43);
        bool reverse_choice = true;
        std::uint64_t reverse_receive = 0;
        ot.iknp_reversed->recv_cot(
            &reverse_receive, &reverse_choice, 1, 43);
        for (int round = 0; round < kRounds; ++round) {
            std::array<std::array<std::uint8_t, kChoices>, kLength> storage{};
            std::array<std::uint8_t *, kLength> messages{};
            for (int index = 0; index < kLength; ++index) {
                messages[index] = storage[index].data();
                for (int choice = 0; choice < kChoices; ++choice)
                    storage[index][choice] = Message(round, index, choice);
            }
            ot.kkot[3]->send(messages.data(), kLength, 8);
        }

        Packet packet;
        const auto received = read(result_pipe[0], &packet, sizeof(packet));
        close(result_pipe[0]);
        int status = 0;
        waitpid(child, &status, 0);
        if (received != static_cast<ssize_t>(sizeof(packet)) || !packet.ok ||
            !WIFEXITED(status) || WEXITSTATUS(status) != 0)
            throw std::runtime_error("VOLE OT client failed");
        constexpr std::uint64_t ring_mask =
            (std::uint64_t{1} << 43) - 1;
        if (packet.straight_receive !=
            ((straight_send + straight_correlation) & ring_mask))
            throw std::runtime_error("VOLE straight correlated OT mismatch");
        if (reverse_receive !=
            ((packet.reverse_send + 11) & ring_mask))
            throw std::runtime_error("VOLE reversed correlated OT mismatch");
        if (revealed_greater.data[0] != 0 || packet.comparison != 0)
            throw std::runtime_error("VOLE comparison returned the wrong bit");
        if (revealed_selected.data[0] != (std::uint64_t{1} << 13) ||
            packet.selected != (std::uint64_t{1} << 13))
            throw std::runtime_error(
                "VOLE multiplexer returned the wrong selected value: " +
                std::to_string(packet.selected));
        if (revealed_maximum.data[0] != (std::uint64_t{1} << 13) ||
            packet.maximum != (std::uint64_t{1} << 13))
            throw std::runtime_error(
                "VOLE max returned the wrong value: " +
                std::to_string(packet.maximum));
        if (revealed_product.data[0] != (std::uint64_t{1} << 26) ||
            packet.product != (std::uint64_t{1} << 26))
            throw std::runtime_error(
                "VOLE batched multiplication returned the wrong value: " +
                std::to_string(packet.product));
        const std::array<std::uint8_t, kLength> choices{0, 3, 9, 15};
        for (int round = 0; round < kRounds; ++round) {
            for (int index = 0; index < kLength; ++index) {
                const auto expected = Message(round, index, choices[index]);
                if (packet.values[round][index] != expected)
                    throw std::runtime_error(
                        "VOLE 1-out-of-N OT mismatch at round " +
                        std::to_string(round) + ", index " +
                        std::to_string(index) + ": actual=" +
                        std::to_string(packet.values[round][index]) +
                        ", expected=" + std::to_string(expected));
            }
        }
        std::cout << "HERA_VOLE_OT_STATUS=PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "HERA_VOLE_OT_STATUS=FAIL\n"
                  << "HERA_VOLE_OT_ERROR=" << error.what() << '\n';
        return 1;
    }
}
