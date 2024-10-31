package the.sharque;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import the.sharque.nn.model.Model;
import the.sharque.nn.neuron.InputRaw;
import the.sharque.nn.neuron.Neuron;
import the.sharque.nn.neuron.NeuronClassification;
import the.sharque.nn.neuron.NeuronInput;
import the.sharque.nn.neuron.NeuronPerceptron;

public class Main {

    public static void main(String[] args) {

        Model model = buildModel();

        String datasetFile = Objects.requireNonNull(Main.class.getClassLoader().getResource("seeds_dataset.txt"))
                .getPath();
        List<List<String>> dataset = readData(datasetFile);
        Collections.shuffle(dataset);

        int length = 13;

        double[][] train = dataset.stream().limit(length)
                .map(line -> line.stream()
                        .limit(7)
                        .mapToDouble(Double::parseDouble)
                        .toArray()).toArray(double[][]::new);

        double[][] train_result = dataset.stream().limit(length)
                .map(line -> line.stream()
                        .skip(7)
                        .limit(1)
                        .mapToDouble(Double::parseDouble)
                        .map(operand -> operand - 1)
                        .toArray()).toArray(double[][]::new);

        double[][] test = dataset.stream().skip(length)
                .map(line -> line.stream()
                        .limit(7)
                        .mapToDouble(Double::parseDouble)
                        .toArray()).toArray(double[][]::new);

        double[][] test_result = dataset.stream().skip(length)
                .map(line -> line.stream()
                        .skip(7)
                        .limit(1)
                        .mapToDouble(Double::parseDouble)
                        .map(operand -> operand - 1)
                        .toArray()).toArray(double[][]::new);

        double learned;
        String oldLog = null;
        do {
            learned = model.learn(train, train_result, 0.01);
            double check = model.predict(test, test_result);
            String log = String.format("Train %f, Test %f", learned, check);
            if (!log.equals(oldLog)) {
                System.out.println(log);
                oldLog = log;
            }
        } while (learned < 1);
    }

    private static Model buildModel() {
        NeuronInput[] inputs = Stream.generate(InputRaw::new)
                .limit(7)
                .toArray(NeuronInput[]::new);

        Neuron[] class1 = Stream.generate(() -> new NeuronPerceptron(inputs))
                .limit(16)
                .toArray(Neuron[]::new);

        Neuron[] class2 = Stream.generate(() -> new NeuronPerceptron(inputs))
                .limit(16)
                .toArray(Neuron[]::new);

        Neuron[] class3 = Stream.generate(() -> new NeuronPerceptron(inputs))
                .limit(16)
                .toArray(Neuron[]::new);

        Neuron[] layer2 = Stream.generate(() -> new NeuronPerceptron(class1, class2, class3))
                .limit(3)
                .toArray(Neuron[]::new);

        Neuron[] output = new Neuron[]{new NeuronClassification(layer2)};

        return new Model(inputs, output);
    }

    private static List<List<String>> readData(String filename) {
        try (Stream<String> lines = Files.lines(Paths.get(filename))) {
            return lines.map(line -> Arrays.asList(line.split("\t")))
                    .collect(Collectors.toList());
        } catch (IOException e) {
            System.err.println("Error reading data: " + e.getMessage());
            throw new RuntimeException(e);
        }
    }
}
