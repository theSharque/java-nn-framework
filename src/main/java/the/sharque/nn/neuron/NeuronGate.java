package the.sharque.nn.neuron;

import static the.sharque.nn.utils.Utils.MAD_LIMIT;
import static the.sharque.nn.utils.Utils.getRandomValue;
import static the.sharque.nn.utils.Utils.isApplicable;

import java.util.Arrays;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import lombok.Getter;
import lombok.Setter;
import the.sharque.nn.utils.Utils;

public class NeuronGate implements Neuron {

    @Getter
    private double result;
    @Getter
    private double learnedGatePlus;
    @Getter
    private double learnedGateMinus;
    @Getter
    private double[] learnedPlus;
    @Getter
    private double[] learnedMinus;
    @Getter
    private double[] weights;
    @Setter
    private double lowGate;
    private final Lock lock = new ReentrantLock();

    private boolean calculated;
    private Neuron[] inputs;
    private final boolean splittable;
    private boolean showed = false;

    public NeuronGate(boolean splittable, Neuron[]... inputs) {
        this.splittable = splittable;
        this.inputs = Arrays.stream(inputs).flatMap(Stream::of).toArray(Neuron[]::new);
        weights = DoubleStream.generate(Utils::getRandomValue).limit(this.inputs.length).toArray();
        learnedPlus = DoubleStream.generate(() -> 0).limit(this.inputs.length).toArray();
        learnedMinus = DoubleStream.generate(() -> 0).limit(this.inputs.length).toArray();

        lowGate = getRandomValue();
        calculated = false;
    }

    @Override
    public String toString() {
        return String.format("{R: %2.4f (W: [%s]) {I: [%s]}",
                result,
                Arrays.stream(weights).mapToObj(value -> String.format("%.2f", value))
                        .collect(Collectors.joining(", ")),
                Arrays.stream(inputs).map(Object::toString).collect(Collectors.joining(", ")));
    }

    @Override
    public void predict() {
        lock.lock();
        try {
            if (!calculated) {
                double summ = IntStream.range(0, inputs.length).unordered()
                        .mapToDouble(i -> {
                            inputs[i].predict();
                            return inputs[i].getResult() * weights[i];
                        })
                        .sum();

                result = summ >= lowGate ? summ : 0;

                calculated = true;
            }
        } finally {
            lock.unlock();
        }
    }

    @Override
    public void reset() {
        lock.lock();
        if (calculated) {
            IntStream.range(0, inputs.length).forEach(i -> inputs[i].reset());
            calculated = false;
        }
        lock.unlock();
    }

    @Override
    public void learn(double learnRate, double value) {
        lock.lock();
        predict();

        if (result != value) {
            double mass = IntStream.range(0, inputs.length).unordered()
                    .mapToDouble(i -> inputs[i].getResult() * weights[i])
                    .sum();

            double newMass = IntStream.range(0, inputs.length).unordered()
                    .mapToDouble(i -> {
                        inputs[i].learn(learnRate, inputs[i].getResult() * (value / mass) * learnRate);
                        inputs[i].predict();
                        return inputs[i].getResult() * weights[i];
                    }).sum();

            IntStream.range(0, inputs.length).unordered().forEach(i -> {
                if (isApplicable()) {
                    double diff = (newMass - value) * inputs[i].getResult();

                    weights[i] -= diff * learnRate;
                    if (weights[i] > MAD_LIMIT || weights[i] < -MAD_LIMIT) {
                        resetWeights();
                    }

                    if (diff >= 0) {
                        learnedPlus[i] += 1;
                    } else {
                        learnedMinus[i] += 1;
                    }
                }
            });

            if (isApplicable()) {
                if(result != 0) {
                    learnedGatePlus += 1;
                } else {
                    learnedGateMinus += 1;
                }

                lowGate -= (lowGate - result) * learnRate;

                if (lowGate > MAD_LIMIT || lowGate < -MAD_LIMIT) {
                    resetWeights();
                }
            }
        }

        calculated = false;
        lock.unlock();
    }

    @Override
    public void resetLearned() {
        lock.lock();
        showed = false;
        learnedGatePlus = 0;
        learnedGateMinus = 0;
        Arrays.parallelSetAll(learnedPlus, value -> 0);
        Arrays.parallelSetAll(learnedMinus, value -> 0);
        Arrays.stream(inputs).forEach(Neuron::resetLearned);
        lock.unlock();
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
                return String.format("\n%sGP:%5.0f GM:%5.0f L:{ %s }",
                        prefix,
                        learnedGatePlus,
                        learnedGateMinus,
                        IntStream.range(0, inputs.length)
                                .mapToObj(i -> String.format("WP:%5.0f WM:%5.0f", learnedPlus[i], learnedMinus[i]))
                                .collect(Collectors.joining(" | ")));
            } else {
                return String.format("\n%sGP:%5.0f GM:%5.0f L:{ %s }\n%sI:{%s}",
                        prefix,
                        learnedGatePlus,
                        learnedGateMinus,
                        IntStream.range(0, inputs.length)
                                .mapToObj(i -> String.format("WP:%5.0f WM:%5.0f", learnedPlus[i], learnedMinus[i]))
                                .collect(Collectors.joining(" | ")),
                        prefix,
                        inData);
            }
        }
    }

    @Override
    public void shock() {
        lock.lock();

        if (learnedGateMinus == learnedGatePlus) {
            lowGate = getRandomValue();
        }

        IntStream.range(0, weights.length)
                .filter(i -> learnedPlus[i] > 0 && learnedMinus[i] > 0 && learnedPlus[i] == learnedMinus[i])
                .forEach(i -> weights[i] = 0);

        lock.unlock();

//        lock.lock();
//        double biggest = Arrays.stream(learned).sum();
//        IntStream.range(0, learned.length)
//                .filter(i -> learned[i] > 0)
//                .filter(i -> learned[i] == biggest).findFirst()
//                .ifPresent(key -> {
//                    learned = new double[learned.length + 1];
//
//                    Neuron[] newInputs = new Neuron[inputs.length + 1];
//                    double[] newWeights = new double[weights.length + 1];
//
//                    IntStream.range(0, inputs.length).forEach(i -> {
//                        newInputs[i] = inputs[i];
//                        newWeights[i] = weights[i];
//                    });
//
//                    newInputs[inputs.length] = inputs[key];
//                    newWeights[weights.length] = getRandomValue();
//                    weights[key] = getRandomValue();
//
//                    inputs = newInputs;
//                    weights = newWeights;
//                });
//        Arrays.stream(inputs).forEach(Neuron::shock);
//        lock.unlock();
    }

    @Override
    public void resetWeights() {
        weights = DoubleStream.generate(Utils::getRandomValue).limit(weights.length).toArray();
        lowGate = getRandomValue();
    }

    @Override
    public boolean isSplittable() {
        return splittable;
    }
}
