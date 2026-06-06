#ifndef MLP_TUI_APP_H
#define MLP_TUI_APP_H

#include <string>
#include <vector>

#include "Canvas.h"
#include "Terminal.h"

namespace tui
{

class App;

// One screen of the interface.
class View
{
public:
    virtual ~View() {}

    // Called before each frame to move the demo on. Return true while
    // something is animated, so that the next frame comes quickly.
    virtual bool update() { return false; }
    virtual void draw(Canvas& canvas) = 0;
    virtual void key(App& app, int key) = 0;
};

// Stack of views: the top one is shown and gets the keys.
class App
{
public:
    // Smallest terminal the views are designed for.
    static const int MIN_WIDTH = 80;
    static const int MIN_HEIGHT = 24;

    explicit App(Terminal& terminal);
    ~App();

    // Show a new view (the app takes ownership) on top of the current one.
    void push(View* view);
    // Close the current view; the app quits after the last one.
    void pop();
    void quit();

    // Draw and handle keys until quit() or a signal.
    void run();

private:
    App(const App&);
    App& operator=(const App&);

    void frame();
    void showError(const std::string& message);
    void deleteClosed();

    Terminal& m_terminal;
    std::vector<View*> m_views;
    std::vector<View*> m_closed; // deleted once their key() call returned
    Canvas m_canvas;
    Canvas m_screen; // what the terminal shows
    bool m_redraw;
    bool m_quit;
};

// The views, in their own files.
View* createMenu();
View* createMessage(const std::string& title, const std::vector<std::string>& lines);

} // namespace tui

#endif // MLP_TUI_APP_H
