package the.sharque.nn.neuron;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import lombok.Getter;
import the.sharque.nn.utils.MaxIndex;

public class NeuronClassification implements Neuron {

    @Getter
    private double result;

    private final Neuron[] inputs;
    private boolean calculated;

    public NeuronClassification(Neuron[]... inputs) {
        this.inputs = Arrays.stream(inputs).flatMap(Stream::of).toArray(Neuron[]::new);
        calculated = false;
    }

    @Override
    public String toString() {
        return String.format("{C: %1.0f {I: [\n%s]}}",
                result,
                Arrays.stream(inputs).map(Object::toString).collect(Collectors.joining(",\n")));
    }

    @Override
    public void predict() {
        if (!calculated) {
            Arrays.stream(inputs).parallel().forEach(Neuron::predict);

            MaxIndex maxIndex = new MaxIndex();
            IntStream.range(0, inputs.length).sequential().forEach(i -> maxIndex.update(i, inputs[i].getResult()));

            result = maxIndex.getIndex();
            calculated = true;
        }
    }

    @Override
    public void reset() {
        Arrays.stream(inputs).parallel().forEach(Neuron::reset);
        calculated = false;
    }

    @Override
    public void learn(double learnRate, double value) {
        predict();

        if (result != value) {
//            inputs[(int) result].learn(learnRate, inputs[(int) value].getResult());
//            inputs[(int) value].learn(learnRate, inputs[(int) result].getResult());
            inputs[(int) result].learn(learnRate, inputs[(int) result].getResult() - 1);
            inputs[(int) value].learn(learnRate, inputs[(int) value].getResult() + 1);
        }
    }
}
