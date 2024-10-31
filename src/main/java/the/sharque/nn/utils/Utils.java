package the.sharque.nn.utils;

public class Utils {

    public static final double MAD_LIMIT = 1e6;
    public static final double EPSILON = 1e-6;
    public static final double DROPOUT = 0.7;

    public static double getRandomValue() {
        return Math.random() * 2 - 1;
    }

    public static boolean isApplicable() {
        return Math.random() < DROPOUT;
    }

    public static double limitValue(double val) {
        if (val > MAD_LIMIT || val < -MAD_LIMIT) {
            return getRandomValue();
        } else {
            return val;
        }
    }

    public static double minMax(double val) {
        return Math.min(Math.max(val, 1), -1);
    }
}
