package me.gommeantilegit.sonopy;

import org.jetbrains.annotations.NotNull;

/**
 * scipy.fftpack.dct type II Java implementation.
 * Remove {@link NotNull} annotations if not present, or add the jetbrains annotations library to the classpath
 * @author GommeAntiLegit
 */
public class DCT {

    /**
     * Type II - 1D discrete cosine transform according to:<br>
     * <pre>
     *           N-1
     * y[k] = 2* sum x[n]*cos(pi*k*(2n+1)/(2*N)), 0 <= k < N.
     *           n=0
     * </pre>
     *
     * if orthoNorm is true, y[k] is multiplied by a scaling factor f defined as:<br>
     * <pre>
     * f = sqrt(1/(4*N)) if k = 0,
     * f = sqrt(1/(2*N)) otherwise.
     * </pre>
     *
     * @param x input tensor
     * @param orthoNorm multiplies the output with a scaling factor f if set to true.
     * @return the result
     */
    @NotNull
    public static float[] dct(@NotNull float[] x, boolean orthoNorm) {
        float[] y = new float[x.length];
        for (int k = 0; k < x.length; k++) {
            float sum = 0;
            for (int n = 0; n < x.length; n++)
                sum += x[n] * Math.cos(Math.PI * k * (2f * n + 1f) / (2f * x.length)); // fast cos will alter results drastically
            y[k] = 2 * sum;
            if (orthoNorm) {
                // fast Inv sqrt will alter results to an extend
                if (k == 0)
                    y[k] *= Math.sqrt(1f / (4f * x.length));
                else
                    y[k] *= Math.sqrt(1f / (2f * x.length));
            }
        }
        return y;
    }

    /**
     * Type II - 2D discrete cosine transform.
     * Performs {@link #dct(float[], boolean)} on x[0 <= n < N]
     * @param x input
     * @param orthoNorm true if ortho normalization should be performed
     * @return result
     * @see #dct(float[], boolean)
     */
    @NotNull
    public static float[][] dct(@NotNull float[][] x, boolean orthoNorm) {
        float[][] y = new float[x.length][];
        for (int i = 0; i < y.length; i++) {
            y[i] = dct(x[i], orthoNorm);
        }
        return y;
    }

}
