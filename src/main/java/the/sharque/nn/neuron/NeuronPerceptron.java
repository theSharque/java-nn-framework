package the.sharque.nn.neuron;

import static the.sharque.nn.utils.Utils.EPSILON;
import static the.sharque.nn.utils.Utils.MAD_LIMIT;
import static the.sharque.nn.utils.Utils.getRandomValue;
import static the.sharque.nn.utils.Utils.isApplicable;

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
    private double[] learned;
    @Getter
    private double[] weights;
    @Setter
    private double bias;
    private final Object lock = new Object();

    private boolean calculated;
    private Neuron[] inputs;
    private final boolean splittable;
    private boolean showed = false;

    public NeuronPerceptron(boolean splittable, Neuron[]... inputs) {
        this.splittable = splittable;
        this.inputs = Arrays.stream(inputs).parallel().flatMap(Stream::of).toArray(Neuron[]::new);
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
            synchronized (lock) {
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
    }

    @Override
    public void reset() {
        synchronized (lock) {
            IntStream.range(0, inputs.length).parallel().forEach(i -> inputs[i].reset());
            calculated = false;
        }
    }

    @Override
    public void learn(double learnRate, double value) {
        predict();

        if (result != value) {
            double diff = (value - result) / inputs.length;
            synchronized (lock) {
                IntStream.range(0, inputs.length).parallel()
                        .forEach(i -> {
                            double requiredValue = diff + inputs[i].getResult() * weights[i];
                            if (isApplicable()) {
                                learned[i] += 1;

                                weights[i] +=
                                        ((requiredValue / (inputs[i].getResult() + EPSILON)) - weights[i]) * learnRate;
                                if (weights[i] > MAD_LIMIT || weights[i] < -MAD_LIMIT) {
                                    System.out.println("Reset perceptron weight");
                                    resetWeights();
                                }

                                inputs[i].learn(learnRate, diff);
                            }
                        });
                if (isApplicable()) {
                    learnedBias += 1;
                    bias += diff * learnRate;
                    if (bias > MAD_LIMIT || bias < -MAD_LIMIT) {
                        System.out.println("Reset perceptron bias");
                        resetWeights();
                    }
                }
            }
        }
    }

    @Override
    public void resetLearned() {
        synchronized (lock) {
            showed = false;
            learnedBias = 0;
            Arrays.parallelSetAll(learned, value -> 0);
            Arrays.stream(inputs).parallel().forEach(Neuron::resetLearned);
        }
    }

    @Override
    public String getLearning(String prefix) {
        if (showed) {
            return "";
        } else {
            showed = true;

            String inData = Arrays.stream(inputs).map(in -> in.getLearning(prefix + "\t"))
                    .collect(Collectors.joining(""));

            if (inData.isEmpty()) {
                return String.format("\n%sBL:%7.2f L:{ %s }",
                        prefix,
                        learnedBias,
                        IntStream.range(0, inputs.length)
                                .mapToObj(i -> String.format("WL:%7.2f", learned[i]))
                                .collect(Collectors.joining(" | ")));
            } else {
                return String.format("\n%sBL:%7.2f L:{ %s }\n%sI:{%s}",
                        prefix,
                        learnedBias,
                        IntStream.range(0, inputs.length)
                                .mapToObj(i -> String.format("WL:%7.2f", learned[i]))
                                .collect(Collectors.joining(" | ")),
                        prefix,
                        inData);
            }
        }
    }

    @Override
    public void shock() {
        synchronized (lock) {
            double biggest = Arrays.stream(learned).parallel().sum();
            IntStream.range(0, learned.length).parallel()
                    .filter(i -> learned[i] > 0)
                    .filter(i -> learned[i] == biggest).findFirst()
                    .ifPresent(key -> {
                        learned = new double[learned.length + 1];

                        Neuron[] newInputs = new Neuron[inputs.length + 1];
                        double[] newWeights = new double[weights.length + 1];

                        IntStream.range(0, inputs.length).parallel().forEach(i -> {
                            newInputs[i] = inputs[i];
                            newWeights[i] = weights[i];
                        });

                        newInputs[inputs.length] = inputs[key];
                        newWeights[weights.length] = getRandomValue();
                        weights[key] = getRandomValue();

                        inputs = newInputs;
                        weights = newWeights;
                    });
        }

        Arrays.stream(inputs).parallel().forEach(Neuron::shock);
    }

    @Override
    public void resetWeights() {
        weights = DoubleStream.generate(Utils::getRandomValue).limit(weights.length).toArray();
        bias = getRandomValue();
    }

    @Override
    public boolean isSplittable() {
        return splittable;
    }
}
