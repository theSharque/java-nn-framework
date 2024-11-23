package the.sharque.nn.neuron;

import java.util.Arrays;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import lombok.Getter;

public class NeuronClassification implements Neuron {

    @Getter
    private double result;
    private double minResult;

    private final Neuron[] inputs;
    private boolean calculated;
    private final Lock lock = new ReentrantLock();
    private boolean showed = false;

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
        lock.lock();
        try {
            if (!calculated) {
                result = findTop(0, inputs.length);
                calculated = true;
            }
        } finally {
            lock.unlock();
        }
    }

    private int findTop(int from, int to) {
        Arrays.stream(inputs).unordered().forEach(Neuron::predict);

        return IntStream.range(from, to).unordered()
                .reduce((l, r) -> inputs[l].getResult() >= inputs[r].getResult() ? l : r)
                .orElse(0);
    }

    @Override
    public void reset() {
        lock.lock();
        if (calculated) {
            Arrays.stream(inputs).forEach(Neuron::reset);
            calculated = false;
        }
        lock.unlock();
    }

    @Override
    public void learn(double learnRate, double required) {
        lock.lock();
        predict();

        if (result != required) {
            inputs[(int) result].learn(learnRate, inputs[(int) result].getResult() - 1);
            if (result > required) {
                IntStream.range(0, (int) required)
                        .forEach(i -> inputs[i].learn(learnRate, inputs[i].getResult() + 1));
            } else {
                IntStream.range((int) required, inputs.length)
                        .forEach(i -> inputs[i].learn(learnRate, inputs[i].getResult() + 1));
            }
        }

        lock.unlock();
    }

    @Override
    public void resetLearned() {
        showed = false;
        Arrays.stream(inputs).forEach(Neuron::resetLearned);
    }

    @Override
    public String getLearning(String prefix) {
        if (showed) {
            return "";
        } else {
            showed = true;
            return Arrays.stream(inputs)
                    .map(neuron -> neuron.getLearning(prefix))
                    .collect(Collectors.joining(""));
        }
    }

    @Override
    public void shock() {
        lock.lock();
        Arrays.stream(inputs).forEach(Neuron::shock);
        lock.unlock();
    }

    @Override
    public void resetWeights() {
        lock.lock();
        this.calculated = false;
        Arrays.stream(inputs).forEach(Neuron::resetWeights);
        lock.unlock();
    }
}
