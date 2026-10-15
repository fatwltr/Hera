#include "hera/mpc_session.h"
#include "hera/detail/worker_ports.h"

#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <algorithm>
#include <csignal>
#include <iostream>
#include <stdexcept>
#include <string>
#include <set>
#include <limits>
#include <vector>

namespace {

void Require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

void Party(hera::PartyRole role, int port, bool mismatch, bool cancel)
{
    hera::SharedMpcSession session(role, "127.0.0.1", port,
        mismatch && role == hera::PartyRole::kClient ? 2 : 3);
    const auto prepared = session.Prepare();
    if (mismatch) {
        Require(!prepared.ok() && prepared.status().message().find("contracts differ") != std::string::npos,
            "worker mismatch was not rejected before opening extra sockets");
        return;
    }
    Require(prepared.ok(), "cannot prepare MPC pool");
    Require(session.Acquire(0) != session.Acquire(1) && session.Acquire(1) != session.Acquire(2),
        "MPC workers share native protocol state");
    if (cancel) {
        bool rejected = false;
        std::atomic<int> completed{0};
        try {
            session.ForEachRange(3, 1, [&](std::size_t worker, std::size_t, std::size_t) {
                ++completed;
                if (worker == 0) throw std::runtime_error("injected MPC failure");
            });
        } catch (const std::runtime_error &error) {
            rejected = std::string(error.what()) == "injected MPC failure";
        }
        Require(rejected && completed == 3, "MPC failure did not join all scheduled workers");
        return;
    }
    for (const auto size : {std::size_t{1}, std::size_t{2}, std::size_t{7}, std::size_t{65539}}) {
        std::vector<std::size_t> owners(size, 99);
        session.ForEachRange(size, 1, [&](std::size_t worker, std::size_t begin, std::size_t end) {
            for (auto i = begin; i < end; ++i) owners.at(i) = worker;
        });
        const auto active = std::min(size, std::size_t{3});
        std::size_t cursor = 0;
        for (std::size_t worker = 0; worker < active; ++worker) {
            const auto end = cursor + size / active + (worker < size % active ? 1 : 0);
            for (; cursor < end; ++cursor)
                Require(owners[cursor] == worker, "MPC range duplicated, omitted, or reordered a value");
        }
    }
    bool invoked = false;
    session.ForEachRange(0, 1, [&](std::size_t, std::size_t, std::size_t) { invoked = true; });
    Require(!invoked, "empty MPC range invoked native protocol");
    bool rejected = false;
    try { session.Acquire(3); } catch (const std::out_of_range &) { rejected = true; }
    Require(rejected, "out-of-range worker was accepted");
}

void Pair(int port, bool mismatch = false, bool cancel = false)
{
    const auto child = fork();
    Require(child >= 0, "cannot fork MPC test client");
    if (child == 0) {
        alarm(40);
        try { Party(hera::PartyRole::kClient, port, mismatch, cancel); _exit(0); }
        catch (const std::exception &error) { std::cerr << error.what() << '\n'; _exit(2); }
    }
    try { Party(hera::PartyRole::kServer, port, mismatch, cancel); }
    catch (...) { kill(child, SIGTERM); waitpid(child, nullptr, 0); throw; }
    int status{};
    waitpid(child, &status, 0);
    Require(WIFEXITED(status) && WEXITSTATUS(status) == 0, "MPC session test client failed");
}

} // namespace

int main()
{
    try {
        // Check expanded layouts without opening hundreds of expensive OT
        // sessions. Every worker's three sockets must remain disjoint.
        for (const auto workers : {std::size_t{1}, std::size_t{16}, std::size_t{32},
                                  std::size_t{50}, std::size_t{51}, std::size_t{128}}) {
            std::set<int> sockets;
            const int base = 23900;
            const int mpc_base = hera::detail::RuntimeMpcPort(base, workers, workers);
            for (std::size_t i = 0; i < workers; ++i) {
                for (int offset : {0, 50, 100}) {
                    Require(sockets.insert(base + 100 + static_cast<int>(3 * i) + offset).second,
                        "conversion sockets overlap");
                    Require(sockets.insert(mpc_base + static_cast<int>(
                        hera::detail::MpcWorkerPortOffset(i)) + offset).second,
                        "MPC sockets overlap conversion or another worker");
                }
            }
            hera::SharedMpcSession accepted(hera::PartyRole::kServer, "", mpc_base, workers);
            Require(accepted.worker_count() == workers, "expanded worker count was capped");
        }
        for (const auto workers : {std::size_t{0}, std::numeric_limits<std::size_t>::max()}) {
            bool rejected = false;
            try { hera::SharedMpcSession invalid(hera::PartyRole::kServer, "", 23900, workers); }
            catch (const std::invalid_argument &) { rejected = true; }
            Require(rejected, "invalid MPC socket lane count was accepted");
        }
        bool rejected = false;
        try { (void)hera::detail::RuntimeMpcPort(65500, 32, 32); }
        catch (const std::invalid_argument &) { rejected = true; }
        Require(rejected, "worker layout overflowed the TCP port space");
        Pair(23900);
        Pair(24100, true);
        Pair(24300, false, true);
        std::cout << "HERA_MPC_SESSION_STATUS=PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "HERA_MPC_SESSION_STATUS=FAIL " << error.what() << '\n';
        return 1;
    }
}
