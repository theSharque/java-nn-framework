package the.sharque.nn.neuron;

import lombok.Setter;

@Setter
public class InputRaw implements NeuronInput {

    private double data;

    @Override
    public String toString() {
        return "" + data;
    }

    @Override
    public double getResult() {
        return data;
    }
}
