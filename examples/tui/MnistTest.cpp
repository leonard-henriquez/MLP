#include <algorithm>
#include <ctime>
#include <exception>
#include <set>
#include <string>
#include <vector>

#include <dirent.h>

#include "mlp/Image.h"
#include "mlp/Mnist.h"
#include "mlp/Network.h"
#include "mlp/Random.h"

#include "App.h"
#include "Session.h"

namespace tui
{

namespace
{

const char* const DRAWINGS_DIRECTORY = "examples/data/digits";

enum Source
{
    TEST_SET,
    DRAWINGS,
    SOURCE_COUNT
};

// A picture of a digit, as drawn and as the network sees it.
struct Drawing
{
    std::string path;
    mlp::Image original;
    mlp::Image digit; // original.toMnistFormat()
    int label;        // from the first character of the file name, -1 if none
};

std::vector<unsigned char> pixelsOf(const mlp::Image& image)
{
    std::vector<unsigned char> pixels(image.width() * image.height());
    for (std::size_t y = 0; y < image.height(); y++)
    {
        for (std::size_t x = 0; x < image.width(); x++)
            pixels[y * image.width() + x] = image.pixel(x, y);
    }
    return pixels;
}

// The image shrunk to fit in size x size pixels (nearest pixel).
mlp::Image thumbnail(const mlp::Image& image, std::size_t size)
{
    std::size_t longest = std::max(image.width(), image.height());
    std::size_t width = std::max<std::size_t>(1, image.width() * size / longest);
    std::size_t height = std::max<std::size_t>(1, image.height() * size / longest);
    mlp::Image small(width, height);
    for (std::size_t y = 0; y < height; y++)
    {
        for (std::size_t x = 0; x < width; x++)
        {
            unsigned char value = image.pixel(x * image.width() / width, y * image.height() / height);
            small.setPixel(x, y, value);
        }
    }
    return small;
}

// Browse the MNIST test images and the sample drawings, and see what the
// network answers for each of them.
class TestView : public View
{
public:
    explicit TestView(Session& session)
        : m_session(session), m_network(0), m_source(TEST_SET), m_label(-1), m_predicted(0),
          m_random(static_cast<unsigned long>(std::time(0)))
    {
        m_index[TEST_SET] = m_index[DRAWINGS] = 0;
        m_correct[TEST_SET] = m_correct[DRAWINGS] = 0;

        if (m_session.hasMnist())
        {
            try
            {
                m_session.test();
            }
            catch (const std::exception& e)
            {
                m_testError = e.what();
            }
        }
        else
        {
            m_testError = "MNIST is not downloaded: run scripts/download_mnist.sh";
        }
        loadDrawings();
        m_network = m_session.network();
        if (!m_testError.empty())
            m_source = DRAWINGS;
        show();
    }

    void draw(Canvas& canvas)
    {
        header(canvas, "MNIST test");
        Footer footer(canvas);
        if (!m_network)
        {
            drawNoNetwork(canvas);
            footer.key("esc", "back").key("q", "quit");
            return;
        }
        footer.key("←→", "browse").key("r", "random").key("m", "mistake").key("tab", "source");
        footer.key("l", "reload").key("esc", "back").key("q", "quit");

        // Sources and network.
        int x = 1;
        const char* names[] = {"MNIST test set", "Your drawings"};
        for (int s = 0; s < SOURCE_COUNT; s++)
        {
            Style style = s == m_source ? Style(WHITE, BLUE, true) : Style(DEFAULT, FAINT);
            x += canvas.text(x, 2, std::string(" ") + names[s] + " ", style) + 1;
        }
        canvas.text(x + 2, 2, "Network " + topology() + ": " + m_session.origin(), Style(DIM));

        int top = 4, rows = canvas.height() - 1 - top;
        if (count() == 0)
        {
            std::vector<std::string> lines;
            lines.push_back(m_source == TEST_SET ? m_testError : m_drawingsError);
            dialog(canvas, "Nothing to show", lines, YELLOW);
            return;
        }

        int scale = rows >= 35 && canvas.width() >= 100 ? 2 : 1;
        int panelWidth = 28 * scale + 4;
        canvas.box(0, top, panelWidth, 14 * scale + 2, title(), BLUE);
        image(canvas, 2, top + 1, &m_pixels[0], 28, 28, scale);
        canvas.text(2, top + 14 * scale + 2, "Network input, 28x28", Style(DIM), panelWidth - 2);

        x = panelWidth;
        int width = canvas.width() - x;
        drawOutputs(canvas, x, top, width);
        drawVerdict(canvas, x, top + 12, width);

        // The original drawing, before it was cropped, scaled and centred.
        if (m_source == DRAWINGS && rows >= 12 + 7 + 16)
        {
            const Drawing& drawing = m_drawings[m_index[DRAWINGS]];
            mlp::Image small = thumbnail(drawing.original, 28);
            std::string size = number(static_cast<long>(drawing.original.width())) + "x"
                               + number(static_cast<long>(drawing.original.height()));
            canvas.box(x, top + 19, 32, 16, "Original " + size);
            std::vector<unsigned char> pixels = pixelsOf(small);
            image(canvas, x + 2, top + 20, &pixels[0], static_cast<int>(small.width()),
                  static_cast<int>(small.height()), 1);
        }
    }

    void key(App& app, int key)
    {
        m_message.clear();
        if (key == KEY_ESCAPE || key == 'b' || key == KEY_BACKSPACE)
        {
            app.pop();
            return;
        }
        if (!m_network)
            return;

        std::size_t n = count();
        std::size_t& index = m_index[m_source];
        if (key == KEY_TAB || key == 'd')
            m_source = (m_source + 1) % SOURCE_COUNT;
        else if (key == 'l')
            reload();
        else if (n == 0)
            return;
        else if (key == KEY_RIGHT || key == ' ')
            index = (index + 1) % n;
        else if (key == KEY_LEFT)
            index = (index + n - 1) % n;
        else if (key == KEY_PAGE_DOWN)
            index = (index + 100) % n;
        else if (key == KEY_PAGE_UP)
            index = (index + n - 100 % n) % n;
        else if (key == KEY_HOME)
            index = 0;
        else if (key == KEY_END)
            index = n - 1;
        else if (key == 'r')
            index = m_random.index(n);
        else if (key == 'm')
            nextMistake();
        show();
    }

private:
    std::size_t count() const
    {
        if (m_source == DRAWINGS)
            return m_drawings.size();
        return m_testError.empty() ? m_session.test().size() : 0;
    }

    std::string topology() const
    {
        std::string text = number(static_cast<long>(m_network->inputCount()));
        for (std::size_t k = 0; k < m_network->layerCount(); k++)
            text += "-" + number(static_cast<long>(m_network->layer(k).size()));
        return text;
    }

    std::string title() const
    {
        if (m_source == DRAWINGS)
        {
            const std::string& path = m_drawings[m_index[DRAWINGS]].path;
            return path.substr(path.rfind('/') + 1);
        }
        return "Test image " + number(static_cast<long>(m_index[TEST_SET]));
    }

    void loadDrawings()
    {
        DIR* directory = opendir(DRAWINGS_DIRECTORY);
        if (!directory)
        {
            m_drawingsError = std::string("Cannot open ") + DRAWINGS_DIRECTORY
                              + ": run bin/tui from the root of the repository";
            return;
        }
        std::vector<std::string> names;
        while (struct dirent* entry = readdir(directory))
        {
            std::string name = entry->d_name;
            if (name.size() > 4 && (name.substr(name.size() - 4) == ".pgm"
                                    || name.substr(name.size() - 4) == ".ppm"))
                names.push_back(name);
        }
        closedir(directory);
        std::sort(names.begin(), names.end());

        for (std::size_t i = 0; i < names.size(); i++)
        {
            try
            {
                Drawing drawing;
                drawing.path = std::string(DRAWINGS_DIRECTORY) + "/" + names[i];
                drawing.original = mlp::Image::load(drawing.path);
                drawing.digit = drawing.original.toMnistFormat();
                drawing.label = names[i][0] >= '0' && names[i][0] <= '9' ? names[i][0] - '0' : -1;
                m_drawings.push_back(drawing);
            }
            catch (const std::exception& e)
            {
                m_drawingsError = e.what(); // the other drawings are still shown
            }
        }
        if (m_drawings.empty() && m_drawingsError.empty())
            m_drawingsError = std::string("No PGM image in ") + DRAWINGS_DIRECTORY;
    }

    // Pixels and label of image i of the current source.
    void get(std::size_t i, std::vector<unsigned char>& pixels, int& label) const
    {
        if (m_source == DRAWINGS)
        {
            pixels = pixelsOf(m_drawings[i].digit);
            label = m_drawings[i].label;
        }
        else
        {
            const mlp::Mnist& test = m_session.test();
            pixels.assign(test.pixels(i), test.pixels(i) + test.pixelCount());
            label = static_cast<int>(test.label(i));
        }
    }

    std::size_t classify(const std::vector<unsigned char>& pixels,
                         std::vector<double>& outputs) const
    {
        std::vector<double> inputs(pixels.size());
        for (std::size_t p = 0; p < pixels.size(); p++)
            inputs[p] = pixels[p] / 255.0;
        outputs = m_network->predict(inputs);
        return static_cast<std::size_t>(std::max_element(outputs.begin(), outputs.end())
                                        - outputs.begin());
    }

    // Classify the current image and count it in the accuracy, once.
    void show()
    {
        if (!m_network || count() == 0)
            return;
        std::size_t index = m_index[m_source];
        get(index, m_pixels, m_label);
        m_predicted = classify(m_pixels, m_outputs);
        if (m_label >= 0 && m_viewed[m_source].insert(index).second
            && m_predicted == static_cast<std::size_t>(m_label))
            m_correct[m_source]++;
    }

    void nextMistake()
    {
        std::size_t n = count();
        std::vector<unsigned char> pixels;
        std::vector<double> outputs;
        int label;
        for (std::size_t step = 1; step <= n; step++)
        {
            std::size_t i = (m_index[m_source] + step) % n;
            get(i, pixels, label);
            if (label >= 0 && classify(pixels, outputs) != static_cast<std::size_t>(label))
            {
                m_index[m_source] = i;
                return;
            }
        }
        m_message = "No mistake: the network recognises all of them.";
    }

    void reload()
    {
        std::string error;
        if (m_session.loadNetwork(error))
        {
            m_message = std::string("Loaded ") + Session::NETWORK_PATH;
            m_viewed[TEST_SET].clear();
            m_viewed[DRAWINGS].clear();
            m_correct[TEST_SET] = m_correct[DRAWINGS] = 0;
        }
        else
        {
            m_message = error;
        }
        m_network = m_session.network();
    }

    void drawNoNetwork(Canvas& canvas)
    {
        std::vector<std::string> lines;
        lines.push_back("There is no network to test yet. Train one in the MNIST");
        lines.push_back("training demo, or from the command line with bin/mnist_train,");
        lines.push_back(std::string("which saves it to ") + Session::NETWORK_PATH + ".");
        if (!m_session.hasMnist())
        {
            lines.push_back("");
            lines.push_back("Both need the MNIST images first (about 11 MB):");
            lines.push_back("");
            lines.push_back("    scripts/download_mnist.sh");
        }
        dialog(canvas, "No trained network", lines, YELLOW);
    }

    void drawOutputs(Canvas& canvas, int x, int y, int width)
    {
        canvas.box(x, y, width, 12, "Network output");
        bool correct = m_label < 0 || m_predicted == static_cast<std::size_t>(m_label);
        for (int d = 0; d < 10; d++)
        {
            int row = y + 1 + d;
            bool predicted = static_cast<std::size_t>(d) == m_predicted;
            bool isLabel = d == m_label;
            int colour = predicted ? (correct ? GREEN : RED) : (isLabel ? YELLOW : BLUE);
            canvas.text(x + 2, row, number(d), Style(predicted ? colour : DIM, DEFAULT, predicted));
            bar(canvas, x + 4, row, width - 19, m_outputs[d], colour);
            canvas.text(x + width - 14, row, fixed(m_outputs[d], 2),
                        Style(predicted ? DEFAULT : DIM, DEFAULT, predicted));
            if (predicted)
                canvas.text(x + width - 9, row, "◀ guess", Style(colour, DEFAULT, true));
            else if (isLabel)
                canvas.text(x + width - 9, row, "◀ true", Style(YELLOW, DEFAULT, true));
        }
    }

    void drawVerdict(Canvas& canvas, int x, int y, int width)
    {
        canvas.box(x, y, width, 7, "Result");
        std::string guess = number(static_cast<long>(m_predicted));
        std::string verdict = "Read as " + guess;
        int colour = BLUE;
        if (m_label >= 0 && m_predicted == static_cast<std::size_t>(m_label))
        {
            verdict = "✓ Right: it is a " + guess;
            colour = GREEN;
        }
        else if (m_label >= 0)
        {
            verdict = "✗ Wrong: read as " + guess + ", it is a " + number(m_label);
            colour = RED;
        }
        canvas.text(x + 2, y + 1, verdict, Style(colour, DEFAULT, true), width - 4);

        std::string position = "Image " + number(static_cast<long>(m_index[m_source] + 1)) + " of "
                               + number(static_cast<long>(count()));
        canvas.text(x + 2, y + 2, position, Style(DIM));

        std::size_t viewed = m_viewed[m_source].size();
        if (viewed > 0)
        {
            double accuracy = static_cast<double>(m_correct[m_source]) / viewed;
            std::string text = "Viewed " + number(static_cast<long>(viewed)) + ", recognised "
                               + number(static_cast<long>(m_correct[m_source])) + ": ";
            int used = canvas.text(x + 2, y + 3, text, Style(DIM));
            canvas.text(x + 2 + used, y + 3, percent(accuracy), Style(DEFAULT, DEFAULT, true));
            bar(canvas, x + 2, y + 4, width - 4, accuracy, GREEN);
        }
        if (!m_message.empty())
            canvas.text(x + 2, y + 5, m_message, Style(YELLOW), width - 4);
    }

    Session& m_session;
    const mlp::Network* m_network; // owned by the session
    std::string m_testError;       // why the MNIST test set is not available
    std::vector<Drawing> m_drawings;
    std::string m_drawingsError;

    int m_source;
    std::size_t m_index[SOURCE_COUNT];
    std::set<std::size_t> m_viewed[SOURCE_COUNT]; // images counted in the accuracy
    std::size_t m_correct[SOURCE_COUNT];

    // Current image.
    std::vector<unsigned char> m_pixels;
    int m_label;
    std::vector<double> m_outputs;
    std::size_t m_predicted;

    mlp::Random m_random;
    std::string m_message;
};

} // namespace

View* createMnistTest(Session& session)
{
    return new TestView(session);
}

} // namespace tui
