#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlp/Dataset.h"
#include "mlp/Network.h"
#include "mlp/Perceptron.h"

#include "App.h"

namespace tui
{

namespace
{

struct Gate
{
    const char* name;
    const char* path;
    const char* rule;
};

const Gate GATES[] = {
    {"AND", "data/ET3.txt", "1 when its three inputs are 1"},
    {"OR", "data/OU2.txt", "1 when at least one input is 1"},
    {"NOR", "data/NON2.txt", "1 when both inputs are 0"},
    {"XOR", "examples/data/xor2.txt", "1 when the inputs differ (not linearly separable)"},
};
const int GATE_COUNT = sizeof GATES / sizeof GATES[0];

// Same settings as bin/perceptron and bin/mlp.
const unsigned PERCEPTRON_EPOCHS = 1000;
const unsigned NETWORK_EPOCHS = 10000;
const std::size_t HIDDEN = 3;
const double RATE = 0.5;
const double TARGET_ERROR = 0.001;

// Training state of one of the two models.
struct Run
{
    std::vector<double> errors; // after each epoch, starting with epoch 0
    bool done;
    bool learned;
    unsigned epochs;
};

void reset(Run& run, double error)
{
    run.errors.assign(1, error);
    run.done = false;
    run.learned = false;
    run.epochs = 0;
}

// A single perceptron and a network with a hidden layer learn the same
// logic gate side by side, epoch after epoch, with a map of what each one
// answers for every point of the input plane.
class GatesView : public View
{
public:
    GatesView()
        : m_gate(GATE_COUNT - 1), m_perceptron(2), m_network(2), m_epoch(0), m_paused(false),
          m_seed(1), m_slice(1)
    {
        restart();
    }

    bool update()
    {
        if (!m_error.empty() || m_paused || (m_perceptronRun.done && m_networkRun.done))
            return false;

        // Speed up as training goes on: one epoch per frame at first, to see
        // the first steps, then more and more.
        unsigned steps = std::max(1u, m_epoch / 20);
        for (unsigned s = 0; s < steps && !(m_perceptronRun.done && m_networkRun.done); s++)
        {
            m_epoch++;
            stepPerceptron();
            stepNetwork();
        }
        return true;
    }

    void draw(Canvas& canvas)
    {
        header(canvas, "Logic gates: perceptron vs multilayer network");
        drawTabs(canvas);

        if (!m_error.empty())
        {
            std::vector<std::string> lines;
            lines.push_back(m_error);
            lines.push_back("");
            lines.push_back("Run bin/tui from the root of the repository.");
            dialog(canvas, "Cannot load the truth table", lines, RED);
        }
        else
        {
            int width = canvas.width() / 2, height = canvas.height() - 5;
            std::string hidden = number(static_cast<long>(HIDDEN));
            std::string inputs = number(static_cast<long>(m_data.inputCount()));
            drawPanel(canvas, 0, 4, width, height, "Single perceptron",
                      "step activation, perceptron rule", true, m_perceptronRun,
                      PERCEPTRON_EPOCHS, ORANGE);
            drawPanel(canvas, width, 4, canvas.width() - width, height,
                      "Multilayer network " + inputs + "-" + hidden + "-1",
                      "sigmoid, backpropagation, rate " + fixed(RATE, 1), false, m_networkRun,
                      NETWORK_EPOCHS, BLUE);
        }

        Footer footer(canvas);
        footer.key("←→", "gate").key("space", m_paused ? "resume" : "pause").key("r", "restart");
        if (m_data.inputCount() > 2)
            footer.key("z", "third input");
        footer.key("esc", "back").key("q", "quit");
    }

    void key(App& app, int key)
    {
        if (key == KEY_ESCAPE || key == 'b' || key == KEY_BACKSPACE)
        {
            app.pop();
        }
        else if (key == KEY_LEFT || key == KEY_RIGHT || (key >= '1' && key < '1' + GATE_COUNT))
        {
            if (key == KEY_LEFT)
                m_gate = (m_gate + GATE_COUNT - 1) % GATE_COUNT;
            else if (key == KEY_RIGHT)
                m_gate = (m_gate + 1) % GATE_COUNT;
            else
                m_gate = key - '1';
            restart();
        }
        else if (key == ' ')
        {
            m_paused = !m_paused;
        }
        else if (key == 'r')
        {
            m_seed++; // other initial weights for the network
            restart();
        }
        else if (key == 'z')
        {
            m_slice = 1 - m_slice;
        }
    }

private:
    void restart()
    {
        m_epoch = 0;
        m_paused = false;
        m_error.clear();
        try
        {
            m_data = mlp::Dataset::load(GATES[m_gate].path);
            if (m_data.inputCount() < 2 || m_data.outputCount() != 1)
                throw std::runtime_error(std::string(GATES[m_gate].path) + " is not a logic gate");
        }
        catch (const std::exception& e)
        {
            m_error = e.what();
            return;
        }

        m_perceptron = mlp::Perceptron(m_data.inputCount());
        m_network = mlp::Network(m_data.inputCount(), m_seed);
        m_network.addLayer(HIDDEN).addLayer(1);
        reset(m_perceptronRun, perceptronError());
        reset(m_networkRun, m_network.error(m_data));
    }

    // Share of the examples the perceptron gets wrong (also its mean
    // squared error, since its outputs are 0 or 1).
    double perceptronError() const
    {
        std::size_t wrong = 0;
        for (std::size_t i = 0; i < m_data.size(); i++)
        {
            if (m_perceptron.output(m_data[i].inputs) != m_data[i].targets[0])
                wrong++;
        }
        return static_cast<double>(wrong) / m_data.size();
    }

    void stepPerceptron()
    {
        if (m_perceptronRun.done)
            return;
        // train() stops after an epoch without any mistake.
        bool learned = m_perceptron.train(m_data, 1);
        m_perceptronRun.errors.push_back(perceptronError());
        m_perceptronRun.epochs = m_epoch;
        m_perceptronRun.learned = learned;
        m_perceptronRun.done = learned || m_epoch >= PERCEPTRON_EPOCHS;
    }

    void stepNetwork()
    {
        if (m_networkRun.done)
            return;
        mlp::TrainingOptions options;
        options.learningRate = RATE;
        options.maxEpochs = 1;
        options.targetError = TARGET_ERROR;
        mlp::TrainingResult result = m_network.train(m_data, options);
        m_networkRun.errors.push_back(result.error);
        m_networkRun.epochs = m_epoch;
        m_networkRun.learned = result.converged;
        m_networkRun.done = result.converged || m_epoch >= NETWORK_EPOCHS;
    }

    // Output of one of the models for the inputs (x1, x2), the other
    // inputs being set to the current slice.
    double output(bool perceptron, double x1, double x2) const
    {
        std::vector<double> inputs(m_data.inputCount(), m_slice);
        inputs[0] = x1;
        inputs[1] = x2;
        return perceptron ? m_perceptron.output(inputs) : m_network.predict(inputs)[0];
    }

    bool inSlice(const mlp::Example& example) const
    {
        for (std::size_t i = 2; i < example.inputs.size(); i++)
        {
            if (example.inputs[i] != m_slice)
                return false;
        }
        return true;
    }

    void drawTabs(Canvas& canvas)
    {
        int x = canvas.text(1, 2, "Gate ", Style(DIM)) + 1;
        for (int g = 0; g < GATE_COUNT; g++)
        {
            Style style = g == m_gate ? Style(WHITE, BLUE, true) : Style(DEFAULT, FAINT);
            x += canvas.text(x, 2, std::string(" ") + GATES[g].name + " ", style) + 1;
        }
        canvas.text(x + 1, 2, GATES[m_gate].rule, Style(DIM));
        if (m_paused)
            canvas.text(canvas.width() - 9, 0, " PAUSED ", Style(BLACK, YELLOW, true));

        // Legend of the maps.
        x = 1 + canvas.text(1, 3, "Output", Style(DIM)) + 2;
        const char* levels[] = {"0", "0.5", "1"};
        for (int i = 0; i < 3; i++)
        {
            canvas.text(x, 3, "██", Style(heat(i / 2.0)));
            x += 3 + canvas.text(x + 3, 3, levels[i], Style(DIM)) + 3;
        }
        canvas.text(x + 1, 3, "● should be 1   ○ should be 0", Style(DIM));
    }

    void drawPanel(Canvas& canvas, int x, int y, int width, int height, const std::string& title,
                   const std::string& description, bool perceptron, const Run& run,
                   unsigned maxEpochs, int colour)
    {
        canvas.box(x, y, width, height, title, colour);
        int left = x + 2, inner = width - 4, top = y + 1, rows = height - 2;
        canvas.text(left, top, description, Style(DIM), inner);

        // Epoch and outcome.
        std::string epoch = "Epoch " + number(run.epochs);
        int column = left + canvas.text(left, top + 1, epoch, Style(DEFAULT, DEFAULT, true)) + 2;
        if (run.learned)
            canvas.text(column, top + 1, "✓ learned", Style(GREEN, DEFAULT, true));
        else if (run.done)
            canvas.text(column, top + 1, "✗ gave up after " + number(maxEpochs),
                        Style(RED, DEFAULT, true), inner - (column - left));
        else
            canvas.text(column, top + 1, "learning...", Style(YELLOW));

        // Map of the outputs: size cells wide and size / 2 high, each cell
        // holding two square pixels on top of each other. The error chart
        // below keeps at least 3 rows.
        const int TABLE_WIDTH = 18;
        int size = std::min((inner - TABLE_WIDTH - 3) / 2, rows - 10);
        size = std::max(2, std::min(size, 14)) * 2;
        int mapX = left + 2, mapY = top + 3;
        for (int row = 0; row < size / 2; row++)
        {
            for (int px = 0; px < size; px++)
            {
                double x1 = static_cast<double>(px) / (size - 1);
                double upper = 1.0 - static_cast<double>(2 * row) / (size - 1);
                double lower = 1.0 - static_cast<double>(2 * row + 1) / (size - 1);
                canvas.pixels(mapX + px, mapY + row, heat(output(perceptron, x1, upper)),
                              heat(output(perceptron, x1, lower)));
            }
        }
        canvas.text(left, mapY, "1", Style(DIM));
        canvas.text(left, mapY + size / 2 - 1, "0", Style(DIM));
        canvas.text(left, mapY + size / 4, "x2", Style(DIM));
        canvas.text(mapX, mapY + size / 2, "0", Style(DIM));
        canvas.text(mapX + size / 2 - 1, mapY + size / 2, "x1", Style(DIM));
        canvas.text(mapX + size - 1, mapY + size / 2, "1", Style(DIM));

        // Training examples on the map and in a table.
        int tableX = mapX + size + 3, tableY = mapY;
        std::string inputs = m_data.inputCount() > 2 ? "x1 x2 x3" : "x1 x2";
        canvas.text(tableX, tableY, "  " + inputs + " → out", Style(DIM));
        if (m_data.inputCount() > 2)
            canvas.text(tableX, mapY + size / 2, "x3 = " + number(m_slice) + " (z)", Style(DIM));
        else if (run.done && !run.learned)
            canvas.text(tableX, mapY + size / 2, "no line splits them", Style(RED));
        for (std::size_t i = 0; i < m_data.size(); i++)
        {
            const mlp::Example& example = m_data[i];
            if (!inSlice(example))
                continue;
            double x1 = example.inputs[0], x2 = example.inputs[1];
            double value = output(perceptron, x1, x2);
            bool wantsOne = example.targets[0] > 0.5;
            std::string marker = wantsOne ? "●" : "○";

            int px = static_cast<int>(x1 * (size - 1) + 0.5);
            int py = static_cast<int>((1.0 - x2) * (size - 1) + 0.5);
            canvas.put(mapX + px, mapY + py / 2, marker, Style(WHITE, heat(value), true));

            std::string line = marker + " ";
            for (std::size_t k = 0; k < example.inputs.size(); k++)
                line += " " + number(static_cast<long>(example.inputs[k])) + " ";
            line += "→ " + (perceptron ? number(static_cast<long>(value)) : fixed(value, 2));
            tableY++;
            int used = canvas.text(tableX, tableY, line);
            bool correct = (value > 0.5) == wantsOne;
            canvas.text(tableX + used + 1, tableY, correct ? "✓" : "✗",
                        Style(correct ? GREEN : RED, DEFAULT, true));
        }

        // Error curve over the epochs.
        int chartY = mapY + std::max(size / 2 + 2, tableY - mapY + 2);
        int chartHeight = top + rows - chartY - 2;
        double high = *std::max_element(run.errors.begin(), run.errors.end());
        high = std::max(high, 0.01);
        int used = canvas.text(left, chartY, "Error ", Style(DIM));
        canvas.text(left + used, chartY, fixed(run.errors.back(), 4), Style(colour, DEFAULT, true));
        std::string scale = "max " + fixed(high, 2);
        canvas.text(left + inner - columns(scale), chartY, scale, Style(DIM));
        chart(canvas, left, chartY + 1, inner, chartHeight, run.errors, 0.0, high, colour);
        canvas.text(left, chartY + 1 + chartHeight, "epoch 0", Style(DIM));
        std::string last = number(static_cast<long>(run.errors.size() - 1));
        canvas.text(left + inner - columns(last), chartY + 1 + chartHeight, last, Style(DIM));
    }

    int m_gate;
    mlp::Dataset m_data;
    std::string m_error; // why the truth table could not be loaded
    mlp::Perceptron m_perceptron;
    mlp::Network m_network;
    Run m_perceptronRun;
    Run m_networkRun;
    unsigned m_epoch;
    bool m_paused;
    unsigned long m_seed;
    int m_slice; // value of the third input on the maps
};

} // namespace

View* createGates()
{
    return new GatesView();
}

} // namespace tui
