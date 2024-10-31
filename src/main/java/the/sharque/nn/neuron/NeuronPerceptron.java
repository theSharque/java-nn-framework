package the.sharque.nn.neuron;

import static the.sharque.nn.utils.Utils.EPSILON;
import static the.sharque.nn.utils.Utils.getRandomValue;
import static the.sharque.nn.utils.Utils.isApplicable;
import static the.sharque.nn.utils.Utils.limitValue;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import lombok.Getter;
import lombok.Setter;
import the.sharque.nn.utils.Utils;

public class NeuronPerceptron implements Neuron {

    @Getter
    private double result;
    @Getter
    private double learnedBias;
    @Getter
    private final double[] learned;
    @Getter
    @Setter
    private double[] weights;
    @Setter
    private double bias;

    private boolean calculated;
    private final Neuron[] inputs;

    public NeuronPerceptron(Neuron[]... inputs) {
        this.inputs = Arrays.stream(inputs).flatMap(Stream::of).toArray(Neuron[]::new);
        weights = DoubleStream.generate(Utils::getRandomValue).limit(this.inputs.length).toArray();
        learned = DoubleStream.generate(() -> 0).limit(this.inputs.length).toArray();

        bias = getRandomValue();
        calculated = true;
    }

    @Override
    public String toString() {
        return String.format("{P: %2.4f (W: [%s]) {I: [%s]}",
                result,
                Arrays.stream(weights).mapToObj(value -> String.format("%.2f", value))
                        .collect(Collectors.joining(", ")),
                Arrays.stream(inputs).map(Object::toString).collect(Collectors.joining(", ")));
    }

    @Override
    public void predict() {
        if (!calculated) {
            result = bias + IntStream.range(0, inputs.length).parallel()
                    .mapToDouble(i -> {
                        inputs[i].predict();
                        return inputs[i].getResult() * weights[i];
                    })
                    .reduce(Double::sum)
                    .orElse(0.0);

            calculated = true;
        }
    }

    @Override
    public void reset() {
        IntStream.range(0, inputs.length).parallel().forEach(i -> inputs[i].reset());
        calculated = false;
    }

    @Override
    public void learn(double learnRate, double value) {
        predict();

        if (result != value) {
            double diff = (value - result);
            bias += diff * learnRate;
            bias = limitValue(bias);
            learnedBias += 1;

            double requiredChanges = diff / inputs.length;
            IntStream.range(0, inputs.length).parallel()
                    .filter(i -> isApplicable())
                    .forEach(i -> {
                        double requiredValue = requiredChanges + inputs[i].getResult() * weights[i];
                        weights[i] += ((requiredValue / (inputs[i].getResult() + EPSILON)) - weights[i]) * learnRate;
                        weights[i] = limitValue(weights[i]);
                        learned[i] += 1;

                        inputs[i].learn(learnRate, requiredChanges);
                    });
        }
    }
}
