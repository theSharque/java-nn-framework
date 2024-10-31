package the.sharque.nn.neuron;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Arrays;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class NeuronPerceptronTest {

    NeuronPerceptron perceptron;


    @BeforeEach
    void setUp() {
        double[] input_data = new double[]{2, 2, -2, -2};
        InputRaw[] inputs = Arrays.stream(input_data)
                .mapToObj(value -> {
                    InputRaw input = new InputRaw();
                    input.setData(value);
                    return input;
                })
                .toArray(InputRaw[]::new);

        perceptron = new NeuronPerceptron(inputs);
        double[] weights = new double[]{3, -3, 3, -3};
        perceptron.setWeights(weights);
        perceptron.setBias(0);
    }

    @Test
    void learn_up_positive() {
        perceptron.learn(1, 10 * perceptron.getWeights().length);
        //   R   W   I   C    nW
        // +10  +3  +2  +6 -> +5
        // +10  -3  +2  -6 -> +5
        // +10  +3  -2  -6 -> -5
        // +10  -3  -2  +6 -> -5
        assertEquals(5, perceptron.getWeights()[0]);
        assertEquals(5, perceptron.getWeights()[1]);
        assertEquals(-5, perceptron.getWeights()[2]);
        assertEquals(-5, perceptron.getWeights()[3]);
    }

    @Test
    void learn_up_negative() {
        perceptron.learn(1, -10 * perceptron.getWeights().length);
        // -10  +3  +2  +6 -> -5
        // -10  -3  +2  -6 -> -5
        // -10  +3  -2  -6 -> +5
        // -10  -3  -2  +6 -> +5
        assertEquals(-5, perceptron.getWeights()[0]);
        assertEquals(-5, perceptron.getWeights()[1]);
        assertEquals(5, perceptron.getWeights()[2]);
        assertEquals(5, perceptron.getWeights()[3]);
    }

    @Test
    void learn_down_positive() {
        perceptron.learn(1, perceptron.getWeights().length);

        //   R   W   I   C     nW
        //  +1  +3  +2  +6 -> +0.5
        //  +1  -3  +2  -6 -> +0.5
        //  +1  +3  -2  +6 -> -0.5
        //  +1  -3  -2  -6 -> -0.5

        assertEquals(0.5, perceptron.getWeights()[0]);
        assertEquals(0.5, perceptron.getWeights()[1]);
        assertEquals(-0.5, perceptron.getWeights()[2]);
        assertEquals(-0.5, perceptron.getWeights()[3]);
    }

    @Test
    void learn_down_negative() {
        perceptron.learn(1, -perceptron.getWeights().length);

        //   R   W   I   C     nW
        //  -1  +3  +2  +6 -> -0.5
        //  -1  -3  +2  -6 -> -0.5
        //  -1  +3  -2  +6 -> +0.5
        //  -1  -3  -2  -6 -> +0.5

        assertEquals(-0.5, perceptron.getWeights()[0]);
        assertEquals(-0.5, perceptron.getWeights()[1]);
        assertEquals(0.5, perceptron.getWeights()[2]);
        assertEquals(0.5, perceptron.getWeights()[3]);
    }
}
