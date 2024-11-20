package the.sharque.nn.utils;

public class Utils {

    public static final double MAD_LIMIT = 1e7;
    public static final double EPSILON = 1e-7;
    public static final double DROPOUT = 0.7;

    public static double getRandomValue() {
        return Math.random() - 0.5;
    }

    public static boolean isApplicable() {
        return Math.random() < DROPOUT;
    }

    public static double limitValue(double val) {
        if (val > MAD_LIMIT || val < -MAD_LIMIT) {
            System.out.printf("Mad value %f\n", val);
            return getRandomValue();
        } else {
            return val;
        }
    }
}
