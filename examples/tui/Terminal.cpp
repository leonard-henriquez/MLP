#include "Terminal.h"

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <stdexcept>

#include <sys/ioctl.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

namespace tui
{

namespace
{

// Alternate screen, hidden cursor, no automatic line wrapping.
const char ENTER_SEQUENCE[] = "\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[2J";
const char LEAVE_SEQUENCE[] = "\x1b[0m\x1b[2J\x1b[?7h\x1b[?25h\x1b[?1049l";

// Shared with the signal handlers, hence plain globals.
struct termios savedSettings;
volatile std::sig_atomic_t active = 0;
volatile std::sig_atomic_t pendingSignal = 0;
volatile std::sig_atomic_t resizePending = 0;

const int HANDLED_SIGNALS[] = {SIGINT, SIGTERM, SIGHUP, SIGWINCH, SIGSEGV, SIGBUS, SIGFPE, SIGABRT};
const int SIGNAL_COUNT = sizeof HANDLED_SIGNALS / sizeof HANDLED_SIGNALS[0];
struct sigaction previousActions[SIGNAL_COUNT];

// Only async-signal-safe calls: this also runs from the signal handlers.
void restore()
{
    if (!active)
        return;
    active = 0;
    ssize_t ignored = ::write(STDOUT_FILENO, LEAVE_SEQUENCE, sizeof LEAVE_SEQUENCE - 1);
    (void)ignored;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &savedSettings);
}

void restoreAtExit()
{
    restore();
}

void onSignal(int number)
{
    if (number == SIGWINCH)
    {
        resizePending = 1;
    }
    else if (number == SIGINT || number == SIGTERM || number == SIGHUP)
    {
        // The main loop quits cleanly on the first one; if it does not
        // answer, a second one ends the program at once.
        if (pendingSignal)
        {
            restore();
            _exit(128 + number);
        }
        pendingSignal = number;
    }
    else
    {
        // Crash: give the terminal back, then die the default way.
        restore();
        std::signal(number, SIG_DFL);
        std::raise(number);
    }
}

} // namespace

bool Terminal::available()
{
    return isatty(STDIN_FILENO) && isatty(STDOUT_FILENO);
}

Terminal::Terminal()
{
    if (active)
        throw std::logic_error("the terminal is already in use");
    if (tcgetattr(STDIN_FILENO, &savedSettings) != 0)
        throw std::runtime_error("cannot read the terminal settings");

    // Raw mode: no echo, no line buffering, Ctrl-C and Ctrl-Z read as keys.
    struct termios raw = savedSettings;
    raw.c_iflag &= ~(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    raw.c_cflag |= CS8;
    raw.c_lflag &= ~(ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) != 0)
        throw std::runtime_error("cannot switch the terminal to raw mode");

    active = 1;
    pendingSignal = 0;
    resizePending = 0;
    static bool registered = false;
    if (!registered)
    {
        std::atexit(restoreAtExit);
        registered = true;
    }

    struct sigaction action;
    action.sa_handler = onSignal;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0; // no SA_RESTART: select() returns early on a resize
    for (int i = 0; i < SIGNAL_COUNT; i++)
        sigaction(HANDLED_SIGNALS[i], &action, &previousActions[i]);

    write(ENTER_SEQUENCE);
}

Terminal::~Terminal()
{
    restore();
    for (int i = 0; i < SIGNAL_COUNT; i++)
        sigaction(HANDLED_SIGNALS[i], &previousActions[i], 0);
}

void Terminal::size(int& columns, int& rows) const
{
    struct winsize size;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col > 0 && size.ws_row > 0)
    {
        columns = size.ws_col;
        rows = size.ws_row;
    }
    else
    {
        columns = 80;
        rows = 24;
    }
}

bool Terminal::resized()
{
    if (!resizePending)
        return false;
    resizePending = 0;
    return true;
}

int Terminal::signal() const
{
    return pendingSignal;
}

bool Terminal::fill(int timeoutMs)
{
    fd_set set;
    FD_ZERO(&set);
    FD_SET(STDIN_FILENO, &set);
    struct timeval timeout;
    timeout.tv_sec = timeoutMs / 1000;
    timeout.tv_usec = (timeoutMs % 1000) * 1000;
    if (select(STDIN_FILENO + 1, &set, 0, 0, timeoutMs < 0 ? 0 : &timeout) <= 0)
        return false; // timeout, or interrupted by a signal

    char buffer[256];
    ssize_t count = read(STDIN_FILENO, buffer, sizeof buffer);
    if (count == 0)
        pendingSignal = SIGHUP; // the terminal went away
    if (count <= 0)
        return false;
    m_input.append(buffer, static_cast<std::size_t>(count));
    return true;
}

int Terminal::readKey(int timeoutMs)
{
    if (pendingSignal)
        return KEY_INTERRUPT;
    if (m_input.empty() && !fill(timeoutMs))
        return pendingSignal ? KEY_INTERRUPT : KEY_NONE;

    unsigned char c = static_cast<unsigned char>(m_input[0]);
    if (c != 0x1b)
    {
        m_input.erase(0, 1);
        switch (c)
        {
        case 3:
            return KEY_INTERRUPT;
        case '\r':
        case '\n':
            return KEY_ENTER;
        case '\t':
            return KEY_TAB;
        case 8:
        case 127:
            return KEY_BACKSPACE;
        }
        return c;
    }

    // Escape alone, or the start of a sequence such as "\x1b[A" (up)? The
    // rest of a sequence arrives at once, so wait only a little for it.
    if (m_input.size() == 1)
        fill(30);
    if (m_input.size() == 1 || (m_input[1] != '[' && m_input[1] != 'O'))
    {
        m_input.erase(0, 1);
        return KEY_ESCAPE;
    }

    std::size_t end = 2;
    for (int attempt = 0; attempt < 2; attempt++)
    {
        while (end < m_input.size() && (m_input[end] < 0x40 || m_input[end] > 0x7e))
            end++;
        if (end < m_input.size() || !fill(30))
            break;
    }
    if (end >= m_input.size())
    {
        m_input.clear();
        return KEY_NONE;
    }

    std::string parameters = m_input.substr(2, end - 2);
    char final = m_input[end];
    m_input.erase(0, end + 1);
    switch (final)
    {
    case 'A':
        return KEY_UP;
    case 'B':
        return KEY_DOWN;
    case 'C':
        return KEY_RIGHT;
    case 'D':
        return KEY_LEFT;
    case 'H':
        return KEY_HOME;
    case 'F':
        return KEY_END;
    case '~':
        if (parameters == "1" || parameters == "7")
            return KEY_HOME;
        if (parameters == "4" || parameters == "8")
            return KEY_END;
        if (parameters == "5")
            return KEY_PAGE_UP;
        if (parameters == "6")
            return KEY_PAGE_DOWN;
        break;
    }
    return KEY_NONE;
}

void Terminal::write(const std::string& data)
{
    std::size_t done = 0;
    while (done < data.size())
    {
        ssize_t count = ::write(STDOUT_FILENO, data.data() + done, data.size() - done);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0)
            throw std::runtime_error("cannot write to the terminal");
        done += static_cast<std::size_t>(count);
    }
}

} // namespace tui
