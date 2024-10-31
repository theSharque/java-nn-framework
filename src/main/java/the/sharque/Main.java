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

//        int length = 5;

//        double[][] train = dataset.stream().limit(length)
//                .map(line -> line.stream()
//                        .limit(7)
//                        .mapToDouble(Double::parseDouble)
//                        .toArray()).toArray(double[][]::new);
//
//        double[][] train_result = dataset.stream().limit(length)
//                .map(line -> line.stream()
//                        .skip(7)
//                        .limit(1)
//                        .mapToDouble(Double::parseDouble)
//                        .map(operand -> operand - 1)
//                        .toArray()).toArray(double[][]::new);
//
//        double[][] test = dataset.stream().skip(length)
//                .map(line -> line.stream()
//                        .limit(7)
//                        .mapToDouble(Double::parseDouble)
//                        .toArray()).toArray(double[][]::new);
//
//        double[][] test_result = dataset.stream().skip(length)
//                .map(line -> line.stream()
//                        .skip(7)
//                        .limit(1)
//                        .mapToDouble(Double::parseDouble)
//                        .map(operand -> operand - 1)
//                        .toArray()).toArray(double[][]::new);

        double[][] train = new double[][]{{11.75, 13.52, 0.8082, 5.444, 2.678, 4.378, 5.31}};
        double[][] train_result = new double[][]{{2}};

        double[][] test = new double[][]{{11.75, 13.52, 0.8082, 5.444, 2.678, 4.378, 5.31}};
        double[][] test_result = new double[][]{{2}};

        double learned;
        do {
            learned = model.learn(train, train_result, 0.1);
            double check = model.predict(test, test_result);
            System.out.printf("Train %f, Test %f\n", learned, check);
        } while (learned < 1);
    }

    private static Model buildModel() {
        NeuronInput[] inputs = Stream.generate(InputRaw::new)
                .limit(7)
                .toArray(NeuronInput[]::new);

        Neuron[] class1 = Stream.generate(() -> new NeuronPerceptron(inputs))
                .limit(1)
                .toArray(Neuron[]::new);

        Neuron[] class2 = Stream.generate(() -> new NeuronPerceptron(inputs))
                .limit(1)
                .toArray(Neuron[]::new);

        Neuron[] class3 = Stream.generate(() -> new NeuronPerceptron(inputs))
                .limit(1)
                .toArray(Neuron[]::new);

//        Neuron[] layer2 = Stream.generate(() -> new NeuronPerceptron(class1, class2, class3))
//                .limit(3)
//                .toArray(Neuron[]::new);

        Neuron[] output = new Neuron[]{new NeuronClassification(class1, class2, class3)};

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
