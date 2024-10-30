package the.sharque.nn.utils;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

@NoArgsConstructor
@AllArgsConstructor
@Getter
public class MaxIndex {

    private int index = 0;
    private double value = 0;

    public void update(int index, double value) {
        if (value > this.value) {
            this.value = value;
            this.index = index;
        }
    }
}
