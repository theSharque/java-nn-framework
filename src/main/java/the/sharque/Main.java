package the.sharque;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import the.sharque.nn.model.Model;
import the.sharque.nn.neuron.InputNeuron;
import the.sharque.nn.neuron.InputRaw;
import the.sharque.nn.neuron.Neuron;
import the.sharque.nn.neuron.NeuronClassification;
import the.sharque.nn.neuron.NeuronPerceptron;

public class Main {

    public static void main(String[] args) {

        Model model = buildModel();

        String datasetFile = Objects.requireNonNull(Main.class.getClassLoader().getResource("seeds_dataset.txt"))
                .getPath();
        List<List<String>> dataset = readData(datasetFile);

        Random rnd = new Random(31);
        Collections.shuffle(dataset, rnd);

        int length = 50;

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
        double oldLearned = 0;
        int step = 0;
        final int batchSize = 100;
        do {
            step++;
            learned = model.learn(train, train_result, 0.01);
            model.reset();
            if (step % batchSize == 0) {
                System.out.println(LocalDateTime.now()
                        .format(DateTimeFormatter.ofPattern("HH:mm:ss")) + " Step: " + step);
                model.showLearning();
//                model.shock();
                model.resetLearned();
            }
            double check = model.predict(test, test_result);

            if (learned > oldLearned || step % batchSize == 0) {
                System.out.printf("%s Train %f, Test %f\n",
                        LocalDateTime.now().format(DateTimeFormatter.ofPattern("HH:mm:ss")), learned, check);
                oldLearned = learned;
            }
        } while (learned < 1);
    }

    private static Model buildModel() {
        InputNeuron[] inputs = Stream.generate(InputRaw::new)
                .limit(7)
                .toArray(InputNeuron[]::new);

        Neuron[] layer1 = Stream.generate(() -> new NeuronPerceptron(true, inputs))
                .limit(7)
                .toArray(Neuron[]::new);

//        Neuron[] layer2 = Stream.generate(() -> new NeuronPerceptron(true, layer1))
//                .limit(1)
//                .toArray(Neuron[]::new);

//        Neuron[] class1 = Stream.generate(() -> new NeuronPerceptron(true, inputs))
//                .limit(3)
//                .toArray(Neuron[]::new);
//
//        Neuron[] out1 = Stream.generate(() -> new NeuronPerceptron(true, class1))
//                .limit(1)
//                .toArray(Neuron[]::new);
//
//        Neuron[] class2 = Stream.generate(() -> new NeuronPerceptron(true, inputs))
//                .limit(3)
//                .toArray(Neuron[]::new);
//
//        Neuron[] out2 = Stream.generate(() -> new NeuronPerceptron(true, class2))
//                .limit(1)
//                .toArray(Neuron[]::new);
//
//        Neuron[] class3 = Stream.generate(() -> new NeuronPerceptron(true, inputs))
//                .limit(3)
//                .toArray(Neuron[]::new);
//
//        Neuron[] out3 = Stream.generate(() -> new NeuronPerceptron(true, class3))
//                .limit(1)
//                .toArray(Neuron[]::new);

        Neuron[] layer3 = Stream.generate(() -> new NeuronPerceptron(false, layer1))
                .limit(3)
                .toArray(Neuron[]::new);

        Neuron[] output = new Neuron[]{new NeuronClassification(layer3)};

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
