package the.sharque.nn.neuron;

import static the.sharque.nn.utils.Utils.isApplicable;

import lombok.Setter;

public class InputStable implements InputNeuron {

    @Setter
    private double data;
    private double weight = 1;
    private double learnedWeight = 0;
    private final Object lock = new Object();

    @Override
    public String toString() {
        return "" + data * weight;
    }

    @Override
    public double getResult() {
        return data * weight;
    }

    @Override
    public void learn(double learnRate, double value) {
        synchronized (lock) {
            if (isApplicable()) {
                if (data * weight > 1 || data * weight < -1) {
                    weight = 2 / (data * 2);
                    learnedWeight++;
                }
            }
        }
    }

    @Override
    public void resetLearned() {
        learnedWeight = 0;
    }

    @Override
    public String getLearning(String prefix) {
        return String.format("WI:%7.2f", learnedWeight);
    }
}
