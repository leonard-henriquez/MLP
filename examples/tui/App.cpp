#include "App.h"

#include <algorithm>
#include <exception>

#include <sys/time.h>

namespace tui
{

namespace
{

// Time between two frames of an animation, in milliseconds.
const long FRAME_TIME = 33;

long milliseconds()
{
    struct timeval now;
    gettimeofday(&now, 0);
    return now.tv_sec * 1000L + now.tv_usec / 1000;
}

// Message in a panel, closed by any key.
class MessageView : public View
{
public:
    MessageView(const std::string& title, const std::vector<std::string>& lines)
        : m_title(title), m_lines(lines)
    {
    }

    void draw(Canvas& canvas)
    {
        header(canvas, m_title);
        dialog(canvas, m_title, m_lines, RED);
        Footer(canvas).key("any key", "back").key("q", "quit");
    }

    void key(App& app, int)
    {
        app.pop();
    }

private:
    std::string m_title;
    std::vector<std::string> m_lines;
};

} // namespace

View* createMessage(const std::string& title, const std::vector<std::string>& lines)
{
    return new MessageView(title, lines);
}

App::App(Terminal& terminal) : m_terminal(terminal), m_redraw(true), m_quit(false)
{
}

App::~App()
{
    deleteClosed();
    for (std::size_t i = 0; i < m_views.size(); i++)
        delete m_views[i];
}

void App::push(View* view)
{
    m_views.push_back(view);
}

void App::pop()
{
    if (m_views.empty())
        return;
    m_closed.push_back(m_views.back());
    m_views.pop_back();
}

void App::quit()
{
    m_quit = true;
}

void App::run()
{
    while (!m_quit && !m_views.empty() && !m_terminal.signal())
    {
        long start = milliseconds();
        bool animated = false;
        try
        {
            animated = m_views.back()->update();
        }
        catch (const std::exception& e)
        {
            showError(e.what());
        }
        frame();

        // Handle the keys typed until the next frame is due.
        long timeout = animated ? std::max(0L, FRAME_TIME - (milliseconds() - start)) : 500;
        for (int key = m_terminal.readKey(static_cast<int>(timeout)); key != KEY_NONE;
             key = m_terminal.readKey(0))
        {
            if (key == KEY_INTERRUPT || key == 'q' || key == 'Q')
            {
                m_quit = true;
                break;
            }
            try
            {
                m_views.back()->key(*this, key);
            }
            catch (const std::exception& e)
            {
                showError(e.what());
            }
            deleteClosed();
            if (m_quit || m_views.empty())
                break;
        }
    }
}

void App::frame()
{
    if (m_views.empty())
        return;
    int width, height;
    m_terminal.size(width, height);
    if (m_terminal.resized() || width != m_canvas.width() || height != m_canvas.height())
    {
        m_canvas.resize(width, height);
        m_redraw = true;
    }

    m_canvas.clear();
    if (width < MIN_WIDTH || height < MIN_HEIGHT)
    {
        std::string lines[] = {"Please enlarge the terminal",
                               "to at least " + number(MIN_WIDTH) + "x" + number(MIN_HEIGHT),
                               "(it is " + number(width) + "x" + number(height) + ")"};
        for (int i = 0; i < 3; i++)
        {
            int x = std::max(0, (width - columns(lines[i])) / 2);
            Style style = i == 0 ? Style(YELLOW, DEFAULT, true) : Style();
            m_canvas.text(x, height / 2 - 1 + i, lines[i], style);
        }
    }
    else
    {
        try
        {
            m_views.back()->draw(m_canvas);
        }
        catch (const std::exception& e)
        {
            showError(e.what());
            return;
        }
    }

    m_terminal.write(m_canvas.render(m_redraw ? 0 : &m_screen));
    m_screen = m_canvas;
    m_redraw = false;
}

void App::showError(const std::string& message)
{
    // The view that failed is closed, so that it does not fail again.
    pop();
    std::vector<std::string> lines;
    lines.push_back(message);
    lines.push_back("");
    lines.push_back("Press any key to go back.");
    push(createMessage("Something went wrong", lines));
}

void App::deleteClosed()
{
    for (std::size_t i = 0; i < m_closed.size(); i++)
        delete m_closed[i];
    m_closed.clear();
}

} // namespace tui
