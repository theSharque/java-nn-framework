package the.sharque.nn.neuron;

import static the.sharque.nn.utils.Utils.EPSILON;
import static the.sharque.nn.utils.Utils.MAD_LIMIT;
import static the.sharque.nn.utils.Utils.isApplicable;

import lombok.Setter;

public class InputFlex implements InputNeuron {

    @Setter
    private double data;
    private double weight = 1;
    private double learnedWeight = 0;
    private final Object lock = new Object();
    private boolean showed = false;

    @Override
    public String toString() {
        return "" + data * weight;
    }

    @Override
    public double getResult() {
        return data * weight;
    }

    @Override
    public void learn(double learnRate, double required) {
        synchronized (lock) {
            if (isApplicable()) {
                double requiredValue = (required - data) + data * weight;
                weight += ((requiredValue / (data + EPSILON)) - weight) * learnRate;
                if (weight > MAD_LIMIT || weight < -MAD_LIMIT) {
                    System.out.println("Reset inputFlex weight");
                    resetWeights();
                }

                learnedWeight++;
            }
        }
    }

    @Override
    public void resetLearned() {
        showed = false;
        learnedWeight = 0;
    }

    @Override
    public void resetWeights() {
        synchronized (lock) {
            weight = 1;
        }
    }

    @Override
    public String getLearning(String prefix) {
        if (showed) {
            return "";
        } else {
            return String.format("%sWI:%7.2f", prefix, learnedWeight);
        }
    }
}
