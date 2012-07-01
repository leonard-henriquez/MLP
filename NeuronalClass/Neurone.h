#ifndef NEURONE_H
#define NEURONE_H


class Neurone
{
    public:
        Neurone();
        virtual ~Neurone();

        virtual std::vector<neurone *> GetInputs() = 0;
        virtual void SetInputs(std::vector<neurone *> val) = 0;
        virtual std::vector<neurone *> GetOutputs() = 0;
        virtual void SetOutputs(std::vector<neurone *> val) = 0;
        virtual std::vector<float> GetWeights() = 0;
        virtual void SetWeights(std::vector<float> val) = 0;
        virtual float GetValue() = 0;
        virtual void SetValue(float val) = 0;

    protected:

    private:
        float m_value;
};

class Neurone_I : public Neurone
{
    public:
        Neurone();
        virtual ~Neurone();

        std::vector<neurone *> GetInputs() { return m_inputs; }
        void SetInputs(std::vector<neurone *> val) { m_inputs = val; }
        std::vector<neurone *> GetOutputs() { return m_outputs; }
        void SetOutputs(std::vector<neurone *> val) { m_outputs = val; }
        std::vector<float> GetWeights() { return m_weights; }
        void SetWeights(std::vector<float> val) { m_weights = val; }
        float GetValue() { return m_value; }
        void SetValue(float val) { m_value = val; }

    protected:

    private:
        std::vector<neurone *> m_inputs;
        std::vector<neurone *> m_outputs;
        std::vector<float> m_weights;
        float m_value;
};

#endif // NEURONE_H
