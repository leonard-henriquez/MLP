// Interactive demos of the library in the terminal, drawn with ANSI escape
// sequences and Unicode block characters (no curses library needed).
//
//     tui
//
// Run it from the root of the repository, where it finds data/ and
// examples/data/.

#include <cstdlib>
#include <exception>
#include <iostream>

#include "App.h"
#include "Terminal.h"

int main()
{
    if (!tui::Terminal::available())
    {
        std::cerr << "error: tui is interactive, run it in a terminal"
                     " (standard input and output must be a TTY)\n";
        return EXIT_FAILURE;
    }

    int signal = 0;
    try
    {
        tui::Terminal terminal;
        tui::App app(terminal);
        app.push(tui::createMenu());
        app.run();
        signal = terminal.signal();
    }
    catch (const std::exception& e)
    {
        std::cerr << "error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
    return signal ? 128 + signal : EXIT_SUCCESS;
}
