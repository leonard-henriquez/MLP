#include "Canvas.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace tui
{

namespace
{

// Bytes in a UTF-8 character starting with `lead`.
std::size_t characterLength(unsigned char lead)
{
    if (lead >= 0xf0)
        return 4;
    if (lead >= 0xe0)
        return 3;
    if (lead >= 0xc0)
        return 2;
    return 1;
}

std::string selectGraphics(const Style& style)
{
    std::ostringstream out;
    out << "\x1b[0";
    if (style.bold)
        out << ";1";
    if (style.fg != DEFAULT)
        out << ";38;5;" << style.fg;
    if (style.bg != DEFAULT)
        out << ";48;5;" << style.bg;
    out << 'm';
    return out.str();
}

double clamp(double value, double low, double high)
{
    return std::max(low, std::min(high, value));
}

const char* const EIGHTHS_UP[] = {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
const char* const EIGHTHS_RIGHT[] = {"▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"};

} // namespace

int gray(double level)
{
    // Black, the 24 grays of the palette, then white.
    int n = static_cast<int>(clamp(level, 0.0, 1.0) * 25 + 0.5);
    if (n == 0)
        return BLACK;
    return n == 25 ? WHITE : 231 + n;
}

int heat(double value)
{
    // Interpolate in the 6x6x6 colour cube: blue, dark gray, orange.
    static const double stops[3][3] = {{0, 2, 5}, {1, 1, 1}, {5, 3, 0}};
    double t = clamp(value, 0.0, 1.0) * 2;
    int k = t < 1 ? 0 : 1;
    double u = t - k;
    int rgb[3];
    for (int c = 0; c < 3; c++)
        rgb[c] = static_cast<int>(stops[k][c] + (stops[k + 1][c] - stops[k][c]) * u + 0.5);
    return 16 + 36 * rgb[0] + 6 * rgb[1] + rgb[2];
}

Style::Style(int fg, int bg, bool bold) : fg(fg), bg(bg), bold(bold)
{
}

bool Style::operator==(const Style& other) const
{
    return fg == other.fg && bg == other.bg && bold == other.bold;
}

Canvas::Canvas(int width, int height) : m_width(0), m_height(0)
{
    resize(width, height);
}

void Canvas::resize(int width, int height)
{
    m_width = std::max(width, 0);
    m_height = std::max(height, 0);
    m_cells.assign(static_cast<std::size_t>(m_width * m_height), Cell());
    clear();
}

void Canvas::clear()
{
    for (std::size_t i = 0; i < m_cells.size(); i++)
    {
        m_cells[i].glyph = " ";
        m_cells[i].style = Style();
    }
}

void Canvas::put(int x, int y, const std::string& glyph, const Style& style)
{
    if (x < 0 || y < 0 || x >= m_width || y >= m_height)
        return;
    Cell& cell = m_cells[static_cast<std::size_t>(y * m_width + x)];
    cell.glyph = glyph;
    cell.style = style;
}

int Canvas::text(int x, int y, const std::string& text, const Style& style, int maxWidth)
{
    if (maxWidth < 0)
        maxWidth = m_width - x;
    int column = 0;
    for (std::size_t i = 0; i < text.size() && column < maxWidth;)
    {
        std::size_t length = characterLength(static_cast<unsigned char>(text[i]));
        if (static_cast<unsigned char>(text[i]) >= ' ')
            put(x + column++, y, text.substr(i, length), style);
        i += length;
    }
    return column;
}

void Canvas::fill(int x, int y, int width, int height, const Style& style)
{
    for (int row = y; row < y + height; row++)
    {
        for (int column = x; column < x + width; column++)
            put(column, row, " ", style);
    }
}

void Canvas::box(int x, int y, int width, int height, const std::string& title, int colour)
{
    if (width < 2 || height < 2)
        return;
    Style border(colour);
    for (int column = x + 1; column < x + width - 1; column++)
    {
        put(column, y, "─", border);
        put(column, y + height - 1, "─", border);
    }
    for (int row = y + 1; row < y + height - 1; row++)
    {
        put(x, row, "│", border);
        put(x + width - 1, row, "│", border);
    }
    put(x, y, "╭", border);
    put(x + width - 1, y, "╮", border);
    put(x, y + height - 1, "╰", border);
    put(x + width - 1, y + height - 1, "╯", border);
    if (!title.empty())
        text(x + 2, y, " " + title + " ", Style(DEFAULT, DEFAULT, true), width - 4);
}

void Canvas::pixels(int x, int y, int top, int bottom)
{
    put(x, y, "▀", Style(top, bottom));
}

bool Canvas::sameLine(const Canvas& other, int y) const
{
    for (int x = 0; x < m_width; x++)
    {
        const Cell& a = m_cells[static_cast<std::size_t>(y * m_width + x)];
        const Cell& b = other.m_cells[static_cast<std::size_t>(y * m_width + x)];
        if (a.glyph != b.glyph || !(a.style == b.style))
            return false;
    }
    return true;
}

std::string Canvas::render(const Canvas* previous) const
{
    bool full = !previous || previous->m_width != m_width || previous->m_height != m_height;
    std::string out = full ? "\x1b[0m\x1b[2J" : "";
    Style current;
    bool styled = false;
    for (int y = 0; y < m_height; y++)
    {
        if (!full && sameLine(*previous, y))
            continue;
        std::ostringstream move;
        move << "\x1b[" << y + 1 << ";1H";
        out += move.str();
        for (int x = 0; x < m_width; x++)
        {
            const Cell& cell = m_cells[static_cast<std::size_t>(y * m_width + x)];
            if (!styled || !(cell.style == current))
            {
                out += selectGraphics(cell.style);
                current = cell.style;
                styled = true;
            }
            out += cell.glyph;
        }
    }
    if (styled)
        out += "\x1b[0m";
    return out;
}

int columns(const std::string& text)
{
    int count = 0;
    for (std::size_t i = 0; i < text.size(); count++)
        i += characterLength(static_cast<unsigned char>(text[i]));
    return count;
}

std::string fixed(double value, int decimals)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(decimals) << value;
    return out.str();
}

std::string number(long value)
{
    std::ostringstream out;
    out << value;
    return out.str();
}

std::string percent(double fraction, int decimals)
{
    return fixed(100.0 * fraction, decimals) + "%";
}

void bar(Canvas& canvas, int x, int y, int width, double fraction, int colour)
{
    int eighths = static_cast<int>(clamp(fraction, 0.0, 1.0) * width * 8 + 0.5);
    for (int column = 0; column < width; column++)
    {
        int level = eighths - column * 8;
        if (level >= 8)
            canvas.put(x + column, y, "█", Style(colour));
        else if (level > 0)
            canvas.put(x + column, y, EIGHTHS_RIGHT[level - 1], Style(colour));
        else
            canvas.put(x + column, y, "─", Style(FAINT));
    }
}

void chart(Canvas& canvas, int x, int y, int width, int height, const std::vector<double>& values,
           double low, double high, int colour)
{
    if (values.empty() || width <= 0 || height <= 0 || high <= low)
        return;
    std::size_t n = values.size();
    int count = std::min(width, static_cast<int>(n));
    for (int column = 0; column < count; column++)
    {
        // Mean of the values falling in this column.
        std::size_t first = n * column / count, last = n * (column + 1) / count;
        double sum = 0.0;
        for (std::size_t i = first; i < last; i++)
            sum += values[i];
        double value = sum / (last - first);

        double fraction = clamp((value - low) / (high - low), 0.0, 1.0);
        int eighths = static_cast<int>(fraction * height * 8 + 0.5);
        for (int row = 0; row < height; row++)
        {
            int level = eighths - row * 8;
            if (level > 0)
                canvas.put(x + column, y + height - 1 - row, EIGHTHS_UP[std::min(level, 8) - 1],
                           Style(colour));
        }
    }
}

void image(Canvas& canvas, int x, int y, const unsigned char* pixels, int width, int height,
           int scale)
{
    int pixelRows = height * scale;
    for (int row = 0; 2 * row < pixelRows; row++)
    {
        for (int column = 0; column < width * scale; column++)
        {
            int px = column / scale;
            int top = gray(pixels[(2 * row / scale) * width + px] / 255.0);
            int bottom = 2 * row + 1 < pixelRows
                             ? gray(pixels[((2 * row + 1) / scale) * width + px] / 255.0)
                             : DEFAULT;
            canvas.pixels(x + column, y + row, top, bottom);
        }
    }
}

void header(Canvas& canvas, const std::string& title)
{
    canvas.text(0, 0, " MLP ", Style(WHITE, BLUE, true));
    canvas.text(6, 0, title, Style(DEFAULT, DEFAULT, true));
}

void dialog(Canvas& canvas, const std::string& title, const std::vector<std::string>& lines,
            int colour)
{
    int width = columns(title) + 8;
    for (std::size_t i = 0; i < lines.size(); i++)
        width = std::max(width, columns(lines[i]) + 6);
    width = std::min(width, canvas.width());
    int height = static_cast<int>(lines.size()) + 4;
    int x = (canvas.width() - width) / 2, y = (canvas.height() - height) / 2;

    canvas.fill(x, y, width, height, Style());
    canvas.box(x, y, width, height, title, colour);
    for (std::size_t i = 0; i < lines.size(); i++)
        canvas.text(x + 3, y + 2 + static_cast<int>(i), lines[i], Style(), width - 6);
}

Footer::Footer(Canvas& canvas) : m_canvas(canvas), m_x(1)
{
}

Footer& Footer::key(const std::string& keys, const std::string& action)
{
    int y = m_canvas.height() - 1;
    m_x += m_canvas.text(m_x, y, keys, Style(BLUE, DEFAULT, true)) + 1;
    m_x += m_canvas.text(m_x, y, action, Style(DIM)) + 3;
    return *this;
}

} // namespace tui
