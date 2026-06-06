#ifndef MLP_TUI_CANVAS_H
#define MLP_TUI_CANVAS_H

#include <string>
#include <vector>

namespace tui
{

// Colours of the 256 colour palette, DEFAULT being the terminal's own.
enum Colour
{
    DEFAULT = -1,
    BLACK = 16,
    WHITE = 231,
    BLUE = 39,
    ORANGE = 208,
    GREEN = 41,
    RED = 203,
    YELLOW = 220,
    DIM = 244,
    FAINT = 238
};

// Gray of the 256 colour palette for a level in [0, 1] (0 is black).
int gray(double level);
// Diverging blue (0) - dark gray (0.5) - orange (1) scale, for outputs in [0, 1].
int heat(double value);

struct Style
{
    Style(int fg = DEFAULT, int bg = DEFAULT, bool bold = false);

    bool operator==(const Style& other) const;

    int fg;
    int bg;
    bool bold;
};

// Screen contents in memory: a grid of cells, each holding one character
// of one column. A frame is drawn on a canvas, then turned into a single
// string of ANSI sequences that only updates the lines that changed.
class Canvas
{
public:
    Canvas(int width = 0, int height = 0);

    int width() const { return m_width; }
    int height() const { return m_height; }

    void resize(int width, int height);
    void clear();

    // Cells outside the canvas are ignored, so drawing never has to check.
    void put(int x, int y, const std::string& glyph, const Style& style = Style());
    // Write UTF-8 text clipped to maxWidth columns (to the right edge if
    // negative). Return the number of columns used.
    int text(int x, int y, const std::string& text, const Style& style = Style(),
             int maxWidth = -1);
    void fill(int x, int y, int width, int height, const Style& style);
    // Rounded frame with an optional title on its top edge.
    void box(int x, int y, int width, int height, const std::string& title = "", int colour = DIM);
    // Two square pixels in one cell: the top one as foreground of an upper
    // half block, the bottom one as background.
    void pixels(int x, int y, int top, int bottom);

    // ANSI sequences drawing this canvas over `previous` (which is on the
    // screen), or over anything if previous is 0 or of another size.
    std::string render(const Canvas* previous) const;

private:
    struct Cell
    {
        std::string glyph;
        Style style;
    };

    bool sameLine(const Canvas& other, int y) const;

    int m_width;
    int m_height;
    std::vector<Cell> m_cells;
};

// Number of columns of a UTF-8 string, every character being one column wide.
int columns(const std::string& text);
// Fixed point formatting: fixed(3.14159, 2) is "3.14".
std::string fixed(double value, int decimals);
std::string number(long value);
std::string percent(double fraction, int decimals = 1);

// Horizontal bar of width columns filled at `fraction`, with 1/8 column
// precision, over a faint track.
void bar(Canvas& canvas, int x, int y, int width, double fraction, int colour);
// Column chart of the values between low and high, resampled to the width.
void chart(Canvas& canvas, int x, int y, int width, int height, const std::vector<double>& values,
           double low, double high, int colour);
// Grayscale image (0 is black, 255 white), each pixel scale x scale square
// pixels, i.e. width * scale columns and height * scale / 2 rows.
void image(Canvas& canvas, int x, int y, const unsigned char* pixels, int width, int height,
           int scale);

// Title line at the top of the screen.
void header(Canvas& canvas, const std::string& title);
// Panel in the middle of the screen, for messages and questions.
void dialog(Canvas& canvas, const std::string& title, const std::vector<std::string>& lines,
            int colour = BLUE);

// Bottom line listing the keys of the current view:
//     Footer(canvas).key("space", "pause").key("esc", "back");
class Footer
{
public:
    explicit Footer(Canvas& canvas);

    Footer& key(const std::string& keys, const std::string& action);

private:
    Canvas& m_canvas;
    int m_x;
};

} // namespace tui

#endif // MLP_TUI_CANVAS_H
