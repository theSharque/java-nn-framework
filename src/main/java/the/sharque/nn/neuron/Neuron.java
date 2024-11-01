package the.sharque.nn.neuron;

import java.io.Serializable;

public interface Neuron extends Serializable {

    double getResult();

    default void predict() {
    }

    default void reset() {
    }

    default void learn(double learnRate, double value) {
    }

    default void resetLearned() {
    }

    String getLearning(String prefix);
}
