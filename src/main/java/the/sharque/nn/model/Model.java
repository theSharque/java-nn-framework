package the.sharque.nn.model;

import java.util.Arrays;
import java.util.stream.IntStream;
import lombok.Getter;
import the.sharque.nn.neuron.Neuron;
import the.sharque.nn.neuron.InputNeuron;

public class Model {

    private final InputNeuron[] input;
    @Getter
    private final Neuron[] output;

    public Model(InputNeuron[] input, Neuron[] output) {
        if (input == null || input.length < 1) {
            throw new IllegalArgumentException("Input neurons contain data");
        }

        this.input = input;
        this.output = output;
    }

    public void reset() {
        Arrays.stream(output).parallel().forEach(Neuron::reset);
    }

    public double predict(double[][] data, double[][] test_result) {
        if (data.length < 1) {
            throw new IllegalArgumentException("Input data must contain at least one row");
        }

        if (data[0].length != input.length) {
            throw new IllegalArgumentException("Input size and data size must have the same length");
        }

        double checked = 0;

        for (int i = 0; i < data.length; i++) {
            double[] dataLine = data[i];
            double[] resultLine = test_result[i];

            reset();
            IntStream.range(0, dataLine.length).parallel().forEach(j -> input[j].setData(dataLine[j]));

            Arrays.stream(output).parallel().forEach(Neuron::predict);

            boolean required = IntStream.range(0, resultLine.length).parallel()
                    .anyMatch(j -> resultLine[j] != output[j].getResult());

            if (!required) {
                checked += 1;
            }
        }

        return checked / data.length;
    }

    public double learn(double[][] data, double[][] result, double lr) {
        if (data.length < 1) {
            throw new IllegalArgumentException("Data must contain at least one row");
        }

        if (data.length != result.length) {
            throw new IllegalArgumentException("Data size and result size must have the same length");
        }

        if (data[0].length != input.length) {
            throw new IllegalArgumentException("Input size and data size must have the same length");
        }

        if (result[0].length != output.length) {
            throw new IllegalArgumentException("Output size and result size must have the same length");
        }

        double learned = 0;
        for (int i = 0; i < data.length; i++) {
            double[] dataLine = data[i];
            double[] resultLine = result[i];

            reset();
            IntStream.range(0, dataLine.length).parallel().forEach(j -> input[j].setData(dataLine[j]));

            Arrays.stream(output).parallel().forEach(Neuron::predict);

            boolean required = IntStream.range(0, resultLine.length).parallel()
                    .anyMatch(j -> resultLine[j] != output[j].getResult());

            if (required) {
                IntStream.range(0, resultLine.length).parallel().forEach(j -> output[j].learn(lr, resultLine[j]));
            } else {
                learned += 1;
            }
        }

        return learned / data.length;
    }

    public void resetLearned() {
        Arrays.stream(output).parallel().forEach(Neuron::resetLearned);
    }
}
