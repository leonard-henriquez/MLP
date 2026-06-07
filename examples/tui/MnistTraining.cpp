#include <algorithm>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include <sys/time.h>

#include "mlp/Mnist.h"
#include "mlp/Network.h"
#include "mlp/Random.h"

#include "App.h"
#include "Session.h"

namespace tui
{

namespace
{

// Choices offered before training.
const int VALUE_COUNT = 6;
struct Setting
{
    const char* name;
    double values[VALUE_COUNT];
    int choice; // index of the current value
    int decimals;
};

const Setting SETTINGS[] = {
    {"Hidden neurons", {16, 32, 64, 100, 200, 300}, 3, 0},
    {"Learning rate", {0.01, 0.02, 0.05, 0.1, 0.2, 0.5}, 3, 2},
    {"Epochs", {1, 2, 3, 5, 10, 20}, 2, 0},
    {"Images per epoch", {1000, 5000, 10000, 20000, 30000, 60000}, 5, 0},
};
enum
{
    HIDDEN,
    RATE,
    EPOCHS,
    IMAGES,
    SETTING_COUNT
};

// Same seed as bin/mnist_train, which this demo reproduces step by step.
const unsigned long SEED = 1;
const double TIME_BUDGET = 0.04;      // seconds of work per frame
const std::size_t ERROR_WINDOW = 500; // images per point of the error curve
const std::size_t SAMPLE_SIZE = 1000; // test images of the quick evaluations

double seconds()
{
    struct timeval now;
    gettimeofday(&now, 0);
    return now.tv_sec + now.tv_usec / 1e6;
}

// Test accuracy measured a few images at a time, between frames.
struct Evaluation
{
    bool active;
    bool full;         // all the test images, or every step-th one
    std::size_t step;
    std::size_t next;  // index of the next test image
    std::size_t done;
    std::size_t total;
    std::size_t correct;
};

// Train a new network on MNIST with Network::trainExample(), a few hundred
// images per frame so that the screen stays alive, and watch its error go
// down and its accuracy on the test images go up.
class TrainingView : public View
{
public:
    explicit TrainingView(Session& session)
        : m_session(session), m_state(session.hasMnist() ? SETUP : MISSING),
          m_settings(SETTINGS, SETTINGS + SETTING_COUNT), m_selected(0), m_loadShown(false),
          m_network(1), m_random(SEED), m_epoch(0), m_position(0), m_trained(0), m_errorSum(0.0),
          m_errorCount(0), m_current(0), m_speed(0.0), m_paused(false), m_confirm(false),
          m_messageColour(GREEN)
    {
        m_evaluation.active = false;
    }

    bool update()
    {
        if (m_state == LOADING)
        {
            // Show the loading message for one frame before blocking on it.
            if (m_loadShown)
                load();
            m_loadShown = true;
            return true;
        }
        if (m_state != TRAINING || m_paused || m_confirm)
            return false;

        const mlp::Mnist& training = m_session.training();
        const mlp::Mnist& test = m_session.test();
        double start = seconds(), trainingTime = 0.0;
        std::size_t trained = 0;
        while (m_state == TRAINING && seconds() - start < TIME_BUDGET)
        {
            if (m_evaluation.active)
            {
                evaluate(test, 100);
                continue;
            }
            double before = seconds();
            for (int n = 0; n < 25 && !m_evaluation.active; n++, trained++)
                trainOne(training);
            trainingTime += seconds() - before;
        }
        if (trained > 0 && trainingTime > 0)
        {
            double speed = trained / trainingTime;
            m_speed = m_speed == 0 ? speed : 0.9 * m_speed + 0.1 * speed;
        }
        training.inputs(m_current, m_inputs);
        m_outputs = m_network.predict(m_inputs);
        return true;
    }

    void draw(Canvas& canvas)
    {
        header(canvas, "MNIST training");
        Footer footer(canvas);
        if (m_state == MISSING)
        {
            drawMissing(canvas);
            footer.key("enter", "check again");
        }
        else if (m_state == SETUP)
        {
            drawSetup(canvas);
            footer.key("↑↓", "choose").key("←→", "change").key("enter", "start");
        }
        else if (m_state == LOADING)
        {
            std::vector<std::string> lines;
            lines.push_back("Reading the 60000 training and 10000 test images");
            lines.push_back(std::string("from ") + Session::MNIST_DIRECTORY + "...");
            dialog(canvas, "Loading MNIST", lines);
        }
        else
        {
            drawTraining(canvas);
            if (m_state == TRAINING)
                footer.key("space", m_paused ? "resume" : "pause").key("x", "stop");
            else
                footer.key("n", "new network");
            footer.key("s", "save").key("t", "test it");
        }
        footer.key("esc", "back").key("q", "quit");

        if (m_confirm)
        {
            std::vector<std::string> lines;
            lines.push_back(std::string(Session::NETWORK_PATH) + " already holds a network,");
            lines.push_back("trained earlier by bin/mnist_train or by this demo.");
            lines.push_back("");
            lines.push_back("y  replace it        any other key  keep it");
            dialog(canvas, "Overwrite the saved network?", lines, YELLOW);
        }
    }

    void key(App& app, int key)
    {
        m_message.clear();
        if (m_confirm)
        {
            m_confirm = false;
            if (key == 'y' || key == 'Y')
                write();
            return;
        }
        if (key == KEY_ESCAPE || key == 'b' || key == KEY_BACKSPACE)
        {
            app.pop();
            return;
        }

        if (m_state == MISSING && key == KEY_ENTER)
        {
            if (m_session.hasMnist())
                m_state = SETUP;
        }
        else if (m_state == SETUP)
        {
            Setting& setting = m_settings[m_selected];
            if (key == KEY_UP || key == 'k')
                m_selected = (m_selected + SETTING_COUNT - 1) % SETTING_COUNT;
            else if (key == KEY_DOWN || key == 'j' || key == KEY_TAB)
                m_selected = (m_selected + 1) % SETTING_COUNT;
            else if ((key == KEY_LEFT || key == 'h') && setting.choice > 0)
                setting.choice--;
            else if ((key == KEY_RIGHT || key == 'l') && setting.choice < VALUE_COUNT - 1)
                setting.choice++;
            else if (key == KEY_ENTER || key == ' ')
            {
                m_state = LOADING;
                m_loadShown = false;
            }
        }
        else if (m_state == TRAINING || m_state == DONE)
        {
            if (key == ' ' && m_state == TRAINING)
                m_paused = !m_paused;
            else if (key == 'x' && m_state == TRAINING)
                finish();
            else if (key == 'n' && m_state == DONE)
                m_state = SETUP;
            else if (key == 's')
                save();
            else if (key == 't')
                test(app);
        }
    }

private:
    enum State
    {
        MISSING, // MNIST is not downloaded
        SETUP,
        LOADING,
        TRAINING,
        DONE
    };

    double value(int setting) const
    {
        const Setting& s = m_settings[setting];
        return s.values[s.choice];
    }

    std::string valueText(int setting) const
    {
        return fixed(value(setting), m_settings[setting].decimals);
    }

    std::string topology() const
    {
        return "784-" + valueText(HIDDEN) + "-10";
    }

    void load()
    {
        try
        {
            const mlp::Mnist& training = m_session.training();
            m_session.test();
            start(training);
        }
        catch (const std::exception& e)
        {
            m_state = SETUP;
            m_message = e.what();
            m_messageColour = RED;
        }
    }

    void start(const mlp::Mnist& training)
    {
        m_network = mlp::Network(training.pixelCount(), SEED);
        m_network.addLayer(static_cast<std::size_t>(value(HIDDEN))).addLayer(10);
        m_random.seed(SEED);

        // The first images of the training set, like --limit in bin/mnist_train.
        std::size_t count = std::min(static_cast<std::size_t>(value(IMAGES)), training.size());
        m_order.resize(count);
        for (std::size_t i = 0; i < count; i++)
            m_order[i] = i;
        shuffle();

        m_epoch = 0;
        m_position = 0;
        m_trained = 0;
        m_errorSum = 0.0;
        m_errorCount = 0;
        m_errors.clear();
        m_accuracies.clear();
        m_epochAccuracies.clear();
        m_current = m_order[0];
        m_speed = 0.0;
        m_paused = false;
        m_state = TRAINING;
        // Accuracy before training, about 10%: chance.
        startEvaluation(false);
    }

    void shuffle()
    {
        for (std::size_t i = m_order.size(); i > 1; i--)
            std::swap(m_order[i - 1], m_order[m_random.index(i)]);
    }

    void trainOne(const mlp::Mnist& training)
    {
        m_current = m_order[m_position];
        training.inputs(m_current, m_inputs);
        training.targets(m_current, m_targets);
        m_errorSum += m_network.trainExample(m_inputs, m_targets, value(RATE));
        m_trained++;
        if (++m_errorCount == ERROR_WINDOW)
        {
            m_errors.push_back(m_errorSum / (ERROR_WINDOW * 10.0));
            m_errorSum = 0.0;
            m_errorCount = 0;
        }

        // A quick evaluation ten times per epoch, a full one at the end.
        std::size_t interval = std::max<std::size_t>(m_order.size() / 10, 1000);
        if (++m_position == m_order.size())
            startEvaluation(true);
        else if (m_position % interval == 0)
            startEvaluation(false);
    }

    void startEvaluation(bool full)
    {
        std::size_t size = m_session.test().size();
        m_evaluation.active = true;
        m_evaluation.full = full;
        m_evaluation.step = full ? 1 : std::max<std::size_t>(size / SAMPLE_SIZE, 1);
        m_evaluation.next = 0;
        m_evaluation.done = 0;
        m_evaluation.total = (size + m_evaluation.step - 1) / m_evaluation.step;
        m_evaluation.correct = 0;
    }

    void evaluate(const mlp::Mnist& test, int count)
    {
        Evaluation& e = m_evaluation;
        for (int n = 0; n < count && e.next < test.size(); n++, e.next += e.step, e.done++)
        {
            test.inputs(e.next, m_inputs);
            if (m_network.classify(m_inputs) == test.label(e.next))
                e.correct++;
        }
        if (e.next < test.size())
            return;

        e.active = false;
        double accuracy = e.done ? static_cast<double>(e.correct) / e.done : 0.0;
        m_accuracies.push_back(accuracy);
        if (!e.full)
            return;
        m_epochAccuracies.push_back(accuracy);
        if (++m_epoch >= static_cast<unsigned>(value(EPOCHS)))
        {
            finish();
        }
        else
        {
            m_position = 0;
            shuffle();
        }
    }

    void finish()
    {
        m_state = DONE;
        m_evaluation.active = false;
        std::string origin = "trained in the demo";
        if (!m_accuracies.empty())
            origin += ", " + percent(m_accuracies.back()) + " on the test images";
        m_session.setNetwork(m_network, origin);
    }

    // Try the network in the test view, as it is now.
    void test(App& app)
    {
        if (m_state == TRAINING)
        {
            m_paused = true;
            m_session.setNetwork(m_network, "being trained in the demo");
        }
        app.push(createMnistTest(m_session));
    }

    void save()
    {
        std::ifstream existing(Session::NETWORK_PATH);
        if (existing)
            m_confirm = true;
        else
            write();
    }

    void write()
    {
        try
        {
            m_network.save(Session::NETWORK_PATH);
            m_session.setNetwork(m_network, Session::NETWORK_PATH);
            m_message = std::string("✓ Saved to ") + Session::NETWORK_PATH;
            m_messageColour = GREEN;
        }
        catch (const std::exception& e)
        {
            m_message = e.what();
            m_messageColour = RED;
        }
    }

    void drawMissing(Canvas& canvas)
    {
        std::vector<std::string> lines;
        lines.push_back("This demo trains a network on MNIST, 60000 training and");
        lines.push_back("10000 test images of handwritten digits, 28x28 pixels each.");
        lines.push_back("");
        lines.push_back("Download them first (about 11 MB), from the root of the");
        lines.push_back("repository:");
        lines.push_back("");
        lines.push_back("    scripts/download_mnist.sh");
        lines.push_back("");
        lines.push_back("then press enter.");
        dialog(canvas, "MNIST is not downloaded yet", lines, YELLOW);
    }

    void drawSetup(Canvas& canvas)
    {
        int width = 64, height = 2 * SETTING_COUNT + 9;
        int x = (canvas.width() - width) / 2, y = std::max(2, (canvas.height() - height) / 2);
        canvas.box(x, y, width, height, "New network", BLUE);

        for (int i = 0; i < SETTING_COUNT; i++)
        {
            int row = y + 2 + 2 * i;
            bool selected = i == m_selected;
            int bg = selected ? FAINT : DEFAULT;
            Style arrow(ORANGE, bg, true);
            if (selected)
                canvas.fill(x + 2, row, width - 4, 1, Style(DEFAULT, bg));

            const Setting& setting = m_settings[i];
            std::string text = valueText(i);
            canvas.text(x + 4, row, setting.name, Style(selected ? WHITE : DEFAULT, bg, true));
            canvas.text(x + 30, row, setting.choice > 0 ? "◀" : " ", arrow);
            canvas.text(x + 38 - columns(text) / 2, row, text, Style(selected ? WHITE : BLUE, bg, true));
            canvas.text(x + 45, row, setting.choice < VALUE_COUNT - 1 ? "▶" : " ", arrow);
        }

        int row = y + 3 + 2 * SETTING_COUNT;
        canvas.text(x + 4, row, "784 pixels → " + valueText(HIDDEN) + " sigmoid neurons → 10 outputs",
                    Style(DIM));
        canvas.text(x + 4, row + 1, "Same training as bin/mnist_train, one image at a time.",
                    Style(DIM));
        if (m_message.empty())
            canvas.text(x + 4, row + 3, "Press enter to start.", Style(DEFAULT, DEFAULT, true));
        else
            canvas.text(x + 4, row + 3, m_message, Style(m_messageColour), width - 8);
    }

    void drawProgress(Canvas& canvas, int y)
    {
        std::string label, details;
        double fraction;
        int colour = BLUE;
        if (m_state == DONE)
        {
            bool complete = m_epoch >= static_cast<unsigned>(value(EPOCHS));
            label = complete ? "Done" : "Stopped";
            fraction = complete ? 1.0 : static_cast<double>(m_position) / m_order.size();
            colour = GREEN;
            details = number(static_cast<long>(m_trained)) + " images learned";
        }
        else if (m_evaluation.active && m_evaluation.full)
        {
            label = "Testing";
            fraction = static_cast<double>(m_evaluation.done) / m_evaluation.total;
            colour = GREEN;
            details = number(static_cast<long>(m_evaluation.done)) + "/"
                      + number(static_cast<long>(m_evaluation.total)) + " test images";
        }
        else
        {
            label = "Epoch " + number(m_epoch + 1) + "/" + valueText(EPOCHS);
            fraction = static_cast<double>(m_position) / m_order.size();
            details = number(static_cast<long>(m_position)) + "/"
                      + number(static_cast<long>(m_order.size()));
            if (m_speed > 0)
                details += "  " + number(static_cast<long>(m_speed)) + " images/s";
        }

        int barX = 14, barWidth = canvas.width() - barX - 40;
        canvas.text(1, y, label, Style(DEFAULT, DEFAULT, true));
        bar(canvas, barX, y, barWidth, fraction, colour);
        canvas.text(barX + barWidth + 2, y, percent(fraction, 0), Style(DEFAULT, DEFAULT, true));
        canvas.text(barX + barWidth + 8, y, details, Style(DIM));
    }

    // Panel with the last value of a curve, its scale and its chart.
    void drawCurve(Canvas& canvas, int x, int y, int width, int height, const std::string& title,
                   const std::vector<double>& values, const std::string& last,
                   const std::string& scale, double low, double high, int colour)
    {
        canvas.box(x, y, width, height, title);
        if (values.empty())
            return;
        int used = canvas.text(x + 2, y + 1, last, Style(colour, DEFAULT, true), width - 4);
        if (used + columns(scale) + 1 <= width - 4)
            canvas.text(x + width - 2 - columns(scale), y + 1, scale, Style(DIM));
        chart(canvas, x + 2, y + 2, width - 4, height - 3, values, low, high, colour);
    }

    void drawTraining(Canvas& canvas)
    {
        int width = canvas.width(), height = canvas.height();
        if (!m_message.empty())
        {
            canvas.text(1, 2, m_message, Style(m_messageColour, DEFAULT, true));
        }
        else
        {
            int x = 1 + canvas.text(1, 2, "Network " + topology() + ", rate " + valueText(RATE)
                                             + ", " + number(static_cast<long>(m_order.size()))
                                             + " images per epoch",
                                    Style(DIM));
            // Accuracy on the whole test set after each epoch.
            std::string epochs;
            for (std::size_t i = 0; i < m_epochAccuracies.size(); i++)
                epochs += " " + percent(m_epochAccuracies[i], 2);
            if (!epochs.empty() && x + 20 + columns(epochs) < width)
            {
                canvas.text(width - 1 - columns(epochs) - 18, 2, "Test set by epoch", Style(DIM));
                canvas.text(width - 1 - columns(epochs), 2, epochs, Style(GREEN, DEFAULT, true));
            }
        }
        if (m_paused)
            canvas.text(width - 9, 0, " PAUSED ", Style(BLACK, YELLOW, true));
        drawProgress(canvas, 3);

        // The image being learned, as large as the screen allows.
        const mlp::Mnist& training = m_session.training();
        int top = 5, rows = height - 1 - top;
        int scale = rows >= 32 && width >= 120 ? 2 : 1;
        int imageWidth = 28 * scale + 4;
        unsigned label = training.label(m_current);
        std::string title = "Training image " + number(static_cast<long>(m_current));
        canvas.box(0, top, imageWidth, rows, title);
        image(canvas, 2, top + 1, training.pixels(m_current), 28, 28, scale);
        int line = top + 1 + 14 * scale;
        canvas.text(2, line, "Label ", Style(DIM));
        canvas.text(8, line, number(label), Style(DEFAULT, DEFAULT, true));
        if (m_outputs.size() == 10)
        {
            unsigned guess = static_cast<unsigned>(
                std::max_element(m_outputs.begin(), m_outputs.end()) - m_outputs.begin());
            bool correct = guess == label;
            canvas.text(2, line + 1, "Network says ", Style(DIM));
            canvas.text(15, line + 1, number(guess) + (correct ? " ✓" : " ✗"),
                        Style(correct ? GREEN : RED, DEFAULT, true));
        }

        // Outputs of the network for this image.
        int x = imageWidth, panelWidth = width - imageWidth;
        canvas.box(x, top, panelWidth, 12, "Network output");
        for (int d = 0; d < 10 && m_outputs.size() == 10; d++)
        {
            int row = top + 1 + d;
            bool isLabel = static_cast<unsigned>(d) == label;
            canvas.text(x + 2, row, number(d), Style(isLabel ? DEFAULT : DIM, DEFAULT, isLabel));
            bar(canvas, x + 4, row, panelWidth - 18, m_outputs[d], isLabel ? GREEN : BLUE);
            canvas.text(x + panelWidth - 13, row, fixed(m_outputs[d], 2), Style(DIM));
            if (isLabel)
                canvas.text(x + panelWidth - 8, row, "◀ label", Style(GREEN));
        }

        // Error and accuracy curves.
        int chartTop = top + 12, chartRows = rows - 12, half = panelWidth / 2;
        double high = m_errors.empty() ? 1.0 : *std::max_element(m_errors.begin(), m_errors.end());
        drawCurve(canvas, x, chartTop, half, chartRows, "Training error", m_errors,
                  m_errors.empty() ? "" : fixed(m_errors.back(), 4), "max " + fixed(high, 3), 0.0,
                  high, ORANGE);

        // The scale starts below the accuracies reached after the first
        // evaluation (chance), to show the progress at the end.
        double low = 1.0;
        for (std::size_t i = m_accuracies.size() > 1 ? 1 : 0; i < m_accuracies.size(); i++)
            low = std::min(low, m_accuracies[i]);
        low = std::max(0.0, static_cast<int>(low * 10) / 10.0 - 0.1);
        drawCurve(canvas, x + half, chartTop, panelWidth - half, chartRows, "Test accuracy",
                  m_accuracies, m_accuracies.empty() ? "" : percent(m_accuracies.back(), 2),
                  percent(low, 0) + "-100%", low, 1.0, GREEN);
    }

    Session& m_session;
    State m_state;
    std::vector<Setting> m_settings;
    int m_selected;
    bool m_loadShown;

    mlp::Network m_network;
    mlp::Random m_random;
    std::vector<std::size_t> m_order; // training images in the order of the epoch
    unsigned m_epoch;                 // epochs completed
    std::size_t m_position;           // in m_order
    std::size_t m_trained;            // images learned since the start
    Evaluation m_evaluation;
    std::vector<double> m_inputs, m_targets, m_outputs;

    double m_errorSum;
    std::size_t m_errorCount;
    std::vector<double> m_errors;         // mean squared error per ERROR_WINDOW images
    std::vector<double> m_accuracies;     // after each evaluation
    std::vector<double> m_epochAccuracies; // on the full test set, after each epoch
    std::size_t m_current;                // training image shown
    double m_speed;                       // images per second

    bool m_paused;
    bool m_confirm; // asking before overwriting the saved network
    std::string m_message;
    int m_messageColour;
};

} // namespace

View* createMnistTraining(Session& session)
{
    return new TrainingView(session);
}

} // namespace tui
