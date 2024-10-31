package the.sharque.nn.neuron;

import static the.sharque.nn.utils.Utils.EPSILON;
import static the.sharque.nn.utils.Utils.limitValue;

import lombok.Setter;

public class InputFlex implements InputNeuron {

    @Setter
    private double data;
    private double weight = 1;
    private double learned = 0;

    @Override
    public String toString() {
        return "" + data * weight;
    }

    @Override
    public double getResult() {
        return data + weight;
    }

    @Override
    public void learn(double learnRate, double value) {
        learned += 1;
        weight += limitValue(value * weight * (1 / (EPSILON + data)) * learnRate);
    }

    @Override
    public void resetLearned() {
        learned = 0;
    }
}
