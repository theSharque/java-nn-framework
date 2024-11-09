package the.sharque.nn.neuron;

import java.io.Serializable;

public interface Neuron extends Serializable {

    double getResult();

    String getLearning(String prefix);

    default boolean isSplittable() {
        return false;
    }

    default void predict() {
    }

    default void reset() {
    }

    default void learn(double learnRate, double value) {
    }

    default void resetLearned() {
    }

    default void shock() {
    }

    void resetWeights();
}
