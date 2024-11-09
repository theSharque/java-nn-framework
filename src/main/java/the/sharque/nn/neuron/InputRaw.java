package the.sharque.nn.neuron;

import lombok.Setter;

@Setter
public class InputRaw implements InputNeuron {

    private double data;
    private boolean showed;

    @Override
    public String toString() {
        return "" + data;
    }

    @Override
    public double getResult() {
        showed = false;
        return data;
    }

    @Override
    public String getLearning(String prefix) {
        if (showed) {
            return "";
        } else {
            showed = true;
            return "0";
        }
    }

    @Override
    public void resetWeights() {
    }
}
