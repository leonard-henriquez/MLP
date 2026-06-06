#include <algorithm>
#include <string>
#include <vector>

#include "App.h"

namespace tui
{

namespace
{

// "MLP" in block letters.
const char* const BANNER[] = {
    "█▄ ▄█ █    █▀▀▄",
    "█ ▀ █ █    █▄▄▀",
    "█   █ █▄▄▄ █   ",
};
const int BANNER_HEIGHT = 3;

struct Item
{
    std::string title;
    std::string description;
    View* (*create)();
};

class MenuView : public View
{
public:
    MenuView() : m_selected(0)
    {
        add("Logic gates", "perceptron vs 2-3-1 network on AND, OR, NOR, XOR", createGates);
    }

    void draw(Canvas& canvas)
    {
        int width = 78;
        int x = (canvas.width() - width) / 2;
        int y = std::max(1, (canvas.height() - 13 - static_cast<int>(m_items.size()) * 2) / 2);

        // Banner shaded from blue to orange, column by column.
        for (int row = 0; row < BANNER_HEIGHT; row++)
        {
            std::string line = BANNER[row];
            int count = columns(line);
            std::size_t i = 0;
            for (int column = 0; column < count; column++)
            {
                std::size_t length = 1;
                while (i + length < line.size() && (line[i + length] & 0xc0) == 0x80)
                    length++;
                canvas.put(x + 2 + column, y + row, line.substr(i, length),
                           Style(heat(static_cast<double>(column) / (count - 1)), DEFAULT, true));
                i += length;
            }
        }
        canvas.text(x + 20, y, "Neural networks from scratch", Style(DEFAULT, DEFAULT, true));
        canvas.text(x + 20, y + 1, "C++98, no dependencies", Style(DIM));
        canvas.text(x + 20, y + 2, "Pick a demo and watch it learn", Style(DIM));

        y += BANNER_HEIGHT + 2;
        for (std::size_t i = 0; i <= m_items.size(); i++)
        {
            bool selected = i == m_selected;
            bool last = i == m_items.size();
            std::string key = last ? "q" : number(static_cast<long>(i + 1));
            std::string title = last ? "Quit" : m_items[i].title;
            int row = y + static_cast<int>(i) * 2 + (last ? 1 : 0);

            if (selected)
                canvas.fill(x, row, width, 1, Style(DEFAULT, FAINT));
            int bg = selected ? FAINT : DEFAULT;
            canvas.text(x + 1, row, selected ? "▶" : " ", Style(ORANGE, bg, true));
            canvas.text(x + 3, row, key, Style(BLUE, bg, true));
            canvas.text(x + 6, row, title, Style(selected ? WHITE : DEFAULT, bg, true));
            if (!last)
                canvas.text(x + 24, row, m_items[i].description, Style(selected ? WHITE : DIM, bg),
                            width - 25);
        }

        header(canvas, "Interactive demos");
        Footer(canvas).key("↑↓", "choose").key("enter", "open").key("q", "quit");
    }

    void key(App& app, int key)
    {
        std::size_t count = m_items.size() + 1; // the items and Quit
        if (key == KEY_UP || key == 'k')
            m_selected = (m_selected + count - 1) % count;
        else if (key == KEY_DOWN || key == 'j' || key == KEY_TAB)
            m_selected = (m_selected + 1) % count;
        else if (key == KEY_HOME)
            m_selected = 0;
        else if (key == KEY_END)
            m_selected = count - 1;
        else if (key == KEY_ESCAPE)
            app.quit();
        else if (key == KEY_ENTER || key == ' ')
            open(app, m_selected);
        else if (key >= '1' && key <= '9')
            open(app, static_cast<std::size_t>(key - '1'));
    }

private:
    void add(const std::string& title, const std::string& description, View* (*create)())
    {
        Item item;
        item.title = title;
        item.description = description;
        item.create = create;
        m_items.push_back(item);
    }

    void open(App& app, std::size_t i)
    {
        if (i == m_items.size())
            app.quit();
        else if (i < m_items.size())
            app.push(m_items[i].create());
    }

    std::vector<Item> m_items;
    std::size_t m_selected;
};

} // namespace

View* createMenu()
{
    return new MenuView();
}

} // namespace tui
