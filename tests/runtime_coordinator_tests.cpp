#include "hera/runtime_coordinator.h"

#include <chrono>
#include <atomic>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {

void Require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

void TestNormalStage(int port)
{
    auto server_future = std::async(std::launch::async, [port] {
        return hera::RuntimeCoordinator::Connect(
            hera::PartyRole::kServer, "", port,
            std::chrono::milliseconds(50), std::chrono::seconds(2));
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    auto client = hera::RuntimeCoordinator::Connect(
        hera::PartyRole::kClient, "127.0.0.1", port,
        std::chrono::milliseconds(50), std::chrono::seconds(2));
    auto server = server_future.get();
    Require(client.ok() && server.ok(), "coordinator connection failed");
    Require(client.value()->Start().ok() && server.value()->Start().ok(),
            "coordinator start failed");
    auto a = std::async(std::launch::async, [&] {
        return server.value()->StageComplete(3, 4);
    });
    auto b = client.value()->StageComplete(3, 4);
    Require(a.get().ok() && b.ok(), "matching stage synchronization failed");
    auto finish = std::async(std::launch::async, [&] {
        return server.value()->Finish();
    });
    auto client_finish = client.value()->Finish();
    Require(finish.get().ok() && client_finish.ok(),
            "normal coordinator finish failed");
}

void TestPeerAbort(int port)
{
    auto server_future = std::async(std::launch::async, [port] {
        return hera::RuntimeCoordinator::Connect(
            hera::PartyRole::kServer, "", port,
            std::chrono::milliseconds(50), std::chrono::seconds(2));
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    auto client = hera::RuntimeCoordinator::Connect(
        hera::PartyRole::kClient, "127.0.0.1", port,
        std::chrono::milliseconds(50), std::chrono::seconds(2));
    auto server = server_future.get();
    Require(client.ok() && server.ok(), "abort connection failed");
    std::atomic<int> cancellations{0};
    client.value()->SetCancellationCallback([&] { ++cancellations; });
    Require(client.value()->Start().ok() && server.value()->Start().ok(),
            "abort coordinator start failed");
    server.value()->Abort(hera::Status::Internal("injected stage failure"),
                          2, 1);
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(2);
    while (client.value()->peer_status().ok() &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    Require(client.value()->peer_status().code() ==
                hera::StatusCode::kPeerAborted,
            "peer abort was not propagated");
    Require(cancellations.load() == 1,
            "peer abort did not invoke cancellation callback exactly once");
    client.value()->Cancel();
}

void TestPeerEof(int port)
{
    auto server_future = std::async(std::launch::async, [port] {
        return hera::RuntimeCoordinator::Connect(
            hera::PartyRole::kServer, "", port,
            std::chrono::milliseconds(50), std::chrono::seconds(2));
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    auto client = hera::RuntimeCoordinator::Connect(
        hera::PartyRole::kClient, "127.0.0.1", port,
        std::chrono::milliseconds(50), std::chrono::seconds(2));
    auto server = server_future.get();
    Require(client.ok() && server.ok(), "EOF connection failed");
    Require(client.value()->Start().ok() && server.value()->Start().ok(),
            "EOF coordinator start failed");
    server.value()->Cancel();
    server.value().reset();
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(2);
    while (client.value()->peer_status().ok() &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    Require(client.value()->peer_status().code() ==
                hera::StatusCode::kCancelled,
            "peer EOF did not cancel the coordinator");
    client.value()->Cancel();
}

void TestHeartbeatTimeout(int port)
{
    auto server_future = std::async(std::launch::async, [port] {
        return hera::RuntimeCoordinator::Connect(
            hera::PartyRole::kServer, "", port,
            std::chrono::milliseconds(50), std::chrono::milliseconds(250));
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    auto client = hera::RuntimeCoordinator::Connect(
        hera::PartyRole::kClient, "127.0.0.1", port,
        std::chrono::milliseconds(50), std::chrono::milliseconds(250));
    auto server = server_future.get();
    Require(client.ok() && server.ok(), "timeout connection failed");
    std::atomic<int> cancellations{0};
    client.value()->SetCancellationCallback([&] { ++cancellations; });
    // Leave the peer coordinator unstarted to model a live TCP connection
    // whose process is stopped or whose network path black-holes frames.
    Require(client.value()->Start().ok(), "timeout coordinator start failed");
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(2);
    while (client.value()->peer_status().ok() &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    Require(client.value()->peer_status().code() ==
                hera::StatusCode::kDeadlineExceeded,
            "missing heartbeat did not reach the deadline");
    Require(cancellations.load() == 1,
            "heartbeat timeout did not cancel protocols exactly once");
    client.value()->Cancel();
    server.value()->Cancel();
}

} // namespace

int main()
{
    try {
        TestNormalStage(29321);
        TestPeerAbort(29331);
        TestPeerEof(29341);
        TestHeartbeatTimeout(29351);
        std::cout << "HERA_RUNTIME_COORDINATOR_STATUS=PASS\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "HERA_RUNTIME_COORDINATOR_STATUS=FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
