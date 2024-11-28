package the.sharque.nn.neuron;

import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class NeuronPerceptronTest {

    InputNeuron[] inputs = Stream.generate(InputRaw::new)
            .limit(7)
            .toArray(InputNeuron[]::new);

    NeuronPerceptron neuron = new NeuronPerceptron(true, inputs);

    @BeforeEach
    void setUp() {
        IntStream.range(0, inputs.length).forEach(j -> inputs[j].setData(j));
    }

    @Test
    void predict() {
        neuron.setWeights(new double[]{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        neuron.reset();
        neuron.predict();
        Assertions.assertEquals(0, neuron.getResult());

        neuron.setWeights(new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0});
        neuron.reset();
        neuron.predict();
        Assertions.assertEquals(5, neuron.getResult());
    }

    @Test
    void learn() {
        neuron.reset();
        neuron.setWeights(new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0});

        for (int i = 0; i < 100; i++) {
            neuron.learn(0.01, 10);
            neuron.reset();
            neuron.predict();
            Assertions.assertTrue(neuron.getResult() <= 15);
        }
    }
}
