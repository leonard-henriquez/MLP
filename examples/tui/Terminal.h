#ifndef MLP_TUI_TERMINAL_H
#define MLP_TUI_TERMINAL_H

#include <string>

namespace tui
{

// Values returned by Terminal::readKey() besides plain characters.
enum Key
{
    KEY_NONE = -1,
    KEY_ENTER = 1000,
    KEY_ESCAPE,
    KEY_BACKSPACE,
    KEY_TAB,
    KEY_UP,
    KEY_DOWN,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_HOME,
    KEY_END,
    KEY_PAGE_UP,
    KEY_PAGE_DOWN,
    KEY_INTERRUPT // Ctrl-C, or SIGINT, SIGTERM or SIGHUP received
};

// The terminal in full screen mode: alternate screen, hidden cursor, and
// keys read one at a time without echo, using only ANSI sequences and
// termios. The previous state is restored by the destructor, at exit and
// when the program is killed by a signal.
class Terminal
{
public:
    // True if the standard input and output are both terminals.
    static bool available();

    Terminal();
    ~Terminal();

    // Current size, in columns and rows.
    void size(int& columns, int& rows) const;
    // True once after each change of size (SIGWINCH).
    bool resized();
    // Signal that stopped the program, 0 if none.
    int signal() const;

    // Wait up to timeoutMs milliseconds (forever if negative) for a key.
    // Return KEY_NONE if none came.
    int readKey(int timeoutMs);

    void write(const std::string& data);

private:
    Terminal(const Terminal&);
    Terminal& operator=(const Terminal&);

    bool fill(int timeoutMs);

    std::string m_input; // bytes read but not yet returned as keys
};

} // namespace tui

#endif // MLP_TUI_TERMINAL_H
