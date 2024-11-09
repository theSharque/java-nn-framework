package the.sharque.nn.neuron;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import lombok.Setter;

@Setter
public class InputRaw implements InputNeuron {

    private double data;
    private boolean showed;
    private final Lock lock = new ReentrantLock();

    @Override
    public String toString() {
        return "" + data;
    }

    @Override
    public double getResult() {
        lock.lock();
        try {
            showed = false;
            return data;
        } finally {
            lock.unlock();
        }
    }

    @Override
    public String getLearning(String prefix) {
        lock.lock();
        try {
            if (showed) {
                return "";
            } else {
                showed = true;
                return "0";
            }
        } finally {
            lock.unlock();
        }
    }

    @Override
    public void resetWeights() {
    }
}
