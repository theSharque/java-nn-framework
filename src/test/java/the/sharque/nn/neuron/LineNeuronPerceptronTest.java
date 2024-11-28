package the.sharque.nn.neuron;

import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class LineNeuronPerceptronTest {

    InputNeuron[] inputs = Stream.generate(InputRaw::new)
            .limit(7)
            .toArray(InputNeuron[]::new);

    NeuronPerceptron[] neuron = new NeuronPerceptron[]{new NeuronPerceptron(true, inputs)};
    NeuronPerceptron neuronFirst = neuron[0];
    NeuronPerceptron neuronSecond = new NeuronPerceptron(true, neuron);

    @BeforeEach
    void setUp() {
        IntStream.range(0, inputs.length).forEach(j -> inputs[j].setData(j));
    }

    @Test
    void predict() {
        neuronFirst.setWeights(new double[]{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        neuronSecond.setWeights(new double[]{1.0});
        neuronSecond.reset();
        neuronSecond.predict();
        Assertions.assertEquals(0, neuronSecond.getResult());

        neuronFirst.setWeights(new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0});
        neuronSecond.setWeights(new double[]{1.0});
        neuronSecond.reset();
        neuronSecond.predict();
        Assertions.assertEquals(5, neuronSecond.getResult());
    }

    @Test
    void learn() {
        neuronFirst.setWeights(new double[]{0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0});
        neuronSecond.setWeights(new double[]{1.0});
        neuronSecond.reset();
        neuronSecond.predict();

        for (int i = 0; i < 1000; i++) {
            neuronSecond.learn(0.001, 10);
            neuronSecond.reset();
            neuronSecond.predict();
            Assertions.assertTrue(neuronFirst.getResult() <= 15);
            Assertions.assertTrue(neuronSecond.getResult() <= 15);
        }
    }
}
