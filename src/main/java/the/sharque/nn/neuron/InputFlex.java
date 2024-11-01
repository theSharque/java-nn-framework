package the.sharque.nn.neuron;

import static the.sharque.nn.utils.Utils.EPSILON;
import static the.sharque.nn.utils.Utils.isApplicable;
import static the.sharque.nn.utils.Utils.limitValue;

import lombok.Setter;

public class InputFlex implements InputNeuron {

    @Setter
    private double data;
    private double bias = 1;
    private double learnedBias = 0;
    private final Object lock = new Object();

    @Override
    public String toString() {
        return "" + data * bias;
    }

    @Override
    public double getResult() {
        return data + bias;
    }

    @Override
    public void learn(double learnRate, double value) {
        synchronized (lock) {
            learnedBias += bias - limitValue(value * bias * (1 / (EPSILON + data)) * learnRate);
            if (isApplicable()) {
                bias += limitValue(value * bias * (1 / (EPSILON + data)) * learnRate);
            }
        }
    }

    @Override
    public void resetLearned() {
        learnedBias = 0;
    }

    @Override
    public String getLearning(String prefix) {
        return "";
    }
}
