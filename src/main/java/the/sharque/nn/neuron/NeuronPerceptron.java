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
import the.sharque.nn.utils.Utils;

public class NeuronPerceptron implements Neuron {

    @Getter
    private double result;
    @Getter
    private double learnedBias;
    @Getter
    private final double[] learned;

    private final Neuron[] inputs;
    private final double[] weights;
    private double bias;
    private boolean calculated;

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
                Arrays.stream(weights).mapToObj(value -> String.format("%.2f", value)).collect(Collectors.joining(", ")),
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

        if (result != value && isApplicable()) {
            double diff = (value - result);
            bias += diff * learnRate;
            bias = limitValue(bias);
            learnedBias += 1;

            double requiredValue = diff / inputs.length;
            IntStream.range(0, inputs.length).parallel()
                    .filter(i -> isApplicable())
                    .forEach(i -> {
                        System.out.printf("%d %.2f %.2f %.2f\n", i, requiredValue, weights[i], inputs[i].getResult());

                        //   R   W   I   C
                        // +10  +3  +2  +6 ->
                        // +10  -3  +2  -6 ->
                        // -10  +3  +2  +6 ->
                        // -10  -3  +2  -6 ->

                        // +10  +3  -2  -6 ->
                        // +10  -3  -2  +6 ->
                        // -10  +3  -2  -6 ->
                        // -10  -3  -2  +6 ->

                        weights[i] += requiredValue * weights[i] * (1 / (EPSILON + inputs[i].getResult())) * learnRate;
                        weights[i] = limitValue(weights[i]);
                        learned[i] += 1;

                        inputs[i].learn(learnRate, requiredValue);
                    });
            System.out.print("\n");
        }
    }
}