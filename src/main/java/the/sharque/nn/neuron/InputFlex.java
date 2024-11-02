package the.sharque.nn.neuron;

import static the.sharque.nn.utils.Utils.EPSILON;
import static the.sharque.nn.utils.Utils.isApplicable;
import static the.sharque.nn.utils.Utils.limitValue;

import lombok.Setter;

public class InputFlex implements InputNeuron {

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
                double requiredValue = (value - data) + data * weight;
                weight += ((requiredValue / (data + EPSILON)) - weight) * learnRate;
                weight = limitValue(weight);
                learnedWeight++;
            }
        }
    }

    @Override
    public void resetLearned() {
        learnedWeight = 0;
    }

    @Override
    public String getLearning(String prefix) {
        return String.format("%sWI:%7.2f", prefix, learnedWeight);
    }
}
