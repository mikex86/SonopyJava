package me.gommeantilegit.sonopy;

import org.jetbrains.annotations.NotNull;

/**
 * @author GommeAntiLegit
 */
public class LinearMath {

    /**
     * Return evenly spaced numbers over a specified interval.
     * Returns num evenly spaced samples, calculated over the interval [start, stop].
     * The endpoint of the interval can optionally be excluded.
     */
    @NotNull
    public static float[] linSpace(float start, float stop, int num, boolean endpoint) {
        float k = (stop - start) / (num - (endpoint ? 1 : 0));
        float[] values = new float[num];
        for (int x = 0; x < num; x++) {
            float y = k * x + start;
            values[x] = y;
        }
        return values;
    }
}
