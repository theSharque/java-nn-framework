package the.sharque.nn.neuron;

import lombok.Setter;

@Setter
public class InputRaw implements InputNeuron {

    private double data;

    @Override
    public String toString() {
        return "" + data;
    }

    @Override
    public double getResult() {
        return data;
    }

    @Override
    public String getLearning(String prefix) {
        return "";
    }
}
