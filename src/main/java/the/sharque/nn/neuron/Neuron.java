package the.sharque.nn.neuron;

public interface Neuron {

    double getResult();

    default void predict() {
    }

    default void reset() {
    }

    default void learn(double learnRate, double value) {
    }

    default void resetLearned() {
    }
}
