package the.sharque.nn.neuron;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import lombok.Getter;

public class NeuronClassification implements Neuron {

    @Getter
    private double result;

    private final Neuron[] inputs;
    private boolean calculated;
    private final Object lock = new Object();

    public NeuronClassification(Neuron[]... inputs) {
        this.inputs = Arrays.stream(inputs).parallel().flatMap(Stream::of).toArray(Neuron[]::new);
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
            synchronized (lock) {
                Arrays.stream(inputs).parallel().forEach(Neuron::predict);

                result = IntStream.range(0, inputs.length).parallel()
                        .reduce((l, r) -> inputs[l].getResult() > inputs[r].getResult() ? l : r)
                        .orElse(0);

                calculated = true;
            }
        }
    }

    @Override
    public void reset() {
        synchronized (lock) {
            Arrays.stream(inputs).parallel().forEach(Neuron::reset);
            calculated = false;
        }
    }

    @Override
    public void learn(double learnRate, double value) {
        predict();

        if (result != value) {
            synchronized (lock) {
                inputs[(int) value].learn(learnRate, inputs[(int) value].getResult() + 1);
                inputs[(int) result].learn(learnRate, inputs[(int) result].getResult() - 1);
            }
        }
    }

    @Override
    public void resetLearned() {
        Arrays.stream(inputs).parallel().forEach(Neuron::resetLearned);
    }

    @Override
    public String getLearning(String prefix) {
        return Arrays.stream(inputs)
                .map(neuron -> neuron.getLearning(prefix))
                .collect(Collectors.joining(""));
    }

    @Override
    public void shock() {
        synchronized (lock) {
            Arrays.stream(inputs).parallel().forEach(Neuron::shock);
        }
    }
}
