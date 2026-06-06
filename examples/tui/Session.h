#ifndef MLP_TUI_SESSION_H
#define MLP_TUI_SESSION_H

#include <string>

#include "mlp/Mnist.h"
#include "mlp/Network.h"

namespace tui
{

// What the MNIST views share: the database, loaded on first use, and the
// network to test.
class Session
{
public:
    static const char* const MNIST_DIRECTORY;
    static const char* const NETWORK_PATH;

    Session();
    ~Session();

    // True if the MNIST files are there (see scripts/download_mnist.sh).
    bool hasMnist() const;
    // Throw std::runtime_error if the files cannot be read.
    const mlp::Mnist& training();
    const mlp::Mnist& test();

    // The network trained last in the demo, or else the one saved in
    // NETWORK_PATH; 0 if there is none.
    const mlp::Network* network();
    // Where the network comes from, or why there is none.
    const std::string& origin() const { return m_origin; }
    void setNetwork(const mlp::Network& network, const std::string& origin);
    // Replace the network by the one saved in NETWORK_PATH. Return false,
    // keeping the current network, if it cannot be loaded.
    bool loadNetwork(std::string& error);

private:
    Session(const Session&);
    Session& operator=(const Session&);

    mlp::Mnist* m_training;
    mlp::Mnist* m_test;
    mlp::Network* m_network;
    std::string m_origin;
    bool m_triedFile;
};

} // namespace tui

#endif // MLP_TUI_SESSION_H
