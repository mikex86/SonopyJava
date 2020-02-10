package me.gommeantilegit.sonopy;

import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Java implementation of the me.gommeantilegit.sonopy.Sonopy audio feature extraction library
 * Can be used to run mycroft-precise models from Java Code
 *
 * @author GommeAntiLegit
 */
public class Sonopy {

    private final int audioWindowSize;
    private final int audioWindowHop;
    private final int fftSize;

    /**
     * Pre computed filterbanks. (Stored in instance with dependent parameters to avoid python @lru_cache() like behavior
     */
    @NotNull
    private final float[][] filterbanks;

    public Sonopy(int sampleRate, int audioWindowSize, int audioWindowHop, int fftSize, int numFilters) {
        this.audioWindowSize = audioWindowSize;
        this.audioWindowHop = audioWindowHop;
        this.fftSize = fftSize;
        this.filterbanks = filterbanks(sampleRate, numFilters, fftSize / 2 + 1);
    }

    /**
     * Prevents error on log(0) or log(-1)
     */
    static float safeLog(float x) {
        if (x <= 0) {
            x = Math.ulp(1f);
        }
        return (float) Math.log(x);
    }

    /**
     * 1D array version of {@link #safeLog(float)}
     * Prevents error on log(0) or log(-1).
     * Stores {@link #safeLog(float)} with x[0 <= n < N] in y[n]
     *
     * @return y
     */
    @NotNull
    static float[] safeLog(@NotNull float[] x) {
        float[] y = new float[x.length];
        for (int i = 0; i < y.length; i++) {
            y[i] = safeLog(x[i]);
        }
        return y;
    }

    /**
     * 2D array version of {@link #safeLog(float)}
     * Prevents error on log(0) or log(-1).
     * Stores {@link #safeLog(float[])} with x[0 <= n < N] in y[n]
     *
     * @return y
     */
    @NotNull
    static float[][] safeLog(@NotNull float[][] x) {
        float[][] y = new float[x.length][x[0].length];
        for (int i = 0; i < y.length; i++) {
            y[i] = safeLog(x[i]);
        }
        return y;
    }

    /**
     * Approximately converts hertz to mel scale values.
     * Inverse operation to {@link #melToHertz(float)}
     */
    static float hertzToMels(float f) {
        return 1127f * safeLog(1f + f / 700f);
    }

    /**
     * Approximately converts mel scale values to hertz.
     * Inverse operation to {@link #hertzToMels(float)}
     */
    static float melToHertz(float mel) {
        return (float) (700f * (Math.exp(mel / 1127f) - 1f));
    }

    /**
     * 1D version of {@link #melToHertz(float)}
     * Stores {@link #melToHertz(float)} with mel[0 <= i < N] in out[i].
     *
     * @return out
     */
    @NotNull
    static float[] melToHertz(@NotNull float[] mel) {
        float[] out = new float[mel.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = melToHertz(mel[i]);
        }
        return out;
    }

    /**
     * Push forward duplicate points to prevent useless filters
     */
    @NotNull
    static int[] correctGrid(@NotNull int[] x) {
        int offset = 0;
        int[] indices = new int[x.length];
        for (int i = 1; i < x.length; i++) {
            int val = x[i];
            int prev = x[0] - 1 + val;
            offset = Math.max(0, offset + prev + 1 - val);
            indices[i] = val + offset;
        }
        return indices;
    }

    /**
     * chop_array([1,2,3], 2, 1) -> [[1,2], [2,3]]
     */
    @NotNull
    static float[][] chopArray(@NotNull float[] array, int windowSize, int hopSize) {
        List<float[]> floats = new ArrayList<>(((array.length + 1 - windowSize) / hopSize) + 1);
        for (int i = windowSize; i < array.length + 1; i += hopSize) {
            floats.add(Arrays.copyOfRange(array, i - windowSize, i));
        }
        return floats.toArray(new float[0][0]);
    }

    /**
     * chop_array([1,2,3], 2, 1) -> [[1,2], [2,3]]
     */
    @NotNull
    static int[][] chopArray(@NotNull int[] array, int windowSize, int hopSize) {
        List<int[]> ints = new ArrayList<>(((array.length + 1 - windowSize) / hopSize) + 1);
        for (int i = windowSize; i < array.length + 1; i += hopSize) {
            ints.add(Arrays.copyOfRange(array, i - windowSize, i));
        }
        return ints.toArray(new int[0][0]);
    }

    /**
     * Makes a set of triangle filters focused on #numFilters mel-spaced frequencies
     */
    @NotNull
    public static float[][] filterbanks(int sampleRate, int numFilters, int fftLen) {
        // Grid contains points for left center and right points of filter triangle
        // mels -> hertz -> fft indices
        float[] gridMels = LinearMath.linSpace(hertzToMels(0f), hertzToMels(sampleRate), numFilters + 2, true);
        float[] gridHertz = melToHertz(gridMels);
        int[] gridIndices = getGridIndices(gridHertz, fftLen, sampleRate);
        float[][] banks = new float[numFilters][fftLen];
        int[][] choppedIndices = chopArray(gridIndices, 3, 1);
        for (int i = 0; i < choppedIndices.length; i++) {
            int[] window = choppedIndices[i];
            int left = window[0], middle = window[1], right = window[2];
            float[] temp;
            temp = LinearMath.linSpace(0, 1, middle - left, false);
            for (int j = left, k = 0; j < middle; j++, k++) {
                banks[i][j] = temp[k];
            }
            temp = LinearMath.linSpace(1, 0, right - middle, false);
            for (int j = middle, k = 0; j < right; j++, k++) {
                banks[i][j] = temp[k];
            }
        }
        return banks;
    }

    /**
     * Converts hertz to fft indices
     */
    @NotNull
    static int[] getGridIndices(@NotNull float[] gridHertz, int fftLen, int sampleRate) {
        int[] intTemp = new int[gridHertz.length];
        for (int i = 0; i < gridHertz.length; i++) {
            float val = gridHertz[i] * fftLen / sampleRate;
            intTemp[i] = (int) (val + Math.ulp(val));
        }
        return correctGrid(intTemp);
    }

    /**
     * Calculates power spectrogram
     */
    @NotNull
    public static float[][] powerSpec(@NotNull float[] audio, int audioWindowSize, int audioWindowHop, int fftSize) {
        float[][] frames = chopArray(audio, audioWindowSize, audioWindowHop);
        float[][] out = new float[frames.length][];
        for (int i = 0; i < frames.length; i++) {
            float[][] fftResult = FFT.rfft(frames[i], fftSize);
            float[] real = fftResult[0], imag = fftResult[1];
            assert real.length == imag.length;
            out[i] = new float[real.length];
            for (int j = 0; j < out[i].length; j++) {
                out[i][j] = (real[j] * real[j] + imag[j] * imag[j]) / (float) fftSize;
            }
        }
        return out;
    }

    /**
     * @param x a 2d input array (tensor shape not required)
     * @return the sum of all elements in #x
     */
    @NotNull
    static float[] sum(@NotNull float[][] x) {
        float[] out = new float[x.length];
        for (int i = 0; i < out.length; i++) {
            for (int j = 0; j < x[i].length; j++) {
                out[i] += x[i][j];
            }
        }
        return out;
    }

    /**
     * Calculates dot product for two 2D matrices (Matrix product)
     * Emulates np.dot for 2D arrays as a and b which is equal to np.matmul
     */
    @NotNull
    static float[][] dot(@NotNull float[][] a, @NotNull float[][] b) {
        if (a[0].length != b.length) {
            throw new IllegalArgumentException("Column of this matrix is not equal to row of second matrix!");
        }
        float[][] out = new float[a.length][b[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b[0].length; j++) {
                float sum = 0;
                for (int k = 0; k < b.length; k++) {
                    sum += a[i][k] * b[k][j];
                }
                out[i][j] = sum;
            }
        }
        return out;
    }

    /**
     * Inner product of vectors
     * x.length must be equal to y.length
     */
    static float dot(@NotNull float[] x, @NotNull float[] y) {
        if (x.length != y.length) throw new RuntimeException("Illegal vector dimensions.");
        float sum = 0.0f;
        for (int i = 0; i < x.length; i++)
            sum += x[i] * y[i];
        return sum;
    }

    /**
     * Returns an a transposed version of the array.
     * Equal to ndarray.T
     * @param array tensor shape required
     */
    @NotNull
    static float[][] transpose(@NotNull float[][] array) {
        float[][] transposed = new float[array[0].length][array.length];
        for (int y = 0; y < array.length; y++) {
            for (int x = 0; x < array[0].length; x++) {
                transposed[x][y] = array[y][x];
            }
        }
        return transposed;
    }

    /**
     * Calculates mel frequency cepstrum coefficient spectrogram.
     */
    @NotNull
    public float[][] mfccSpec(@NotNull float[] audio, int numCoeffs) {
        float[][] powers = powerSpec(audio, audioWindowSize, audioWindowHop, fftSize);
        if (powers.length == 0)
            throw new IllegalStateException("powers length 0");

        assert fftSize / 2 + 1 == powers[0].length;
        float[][] filtersTransposed = transpose(this.filterbanks);
        float[][] powerFilterDot = dot(powers, filtersTransposed);
        float[][] mels = safeLog(powerFilterDot);
        float[][] mfccs = DCT.dct(mels, true);
        for (int i = 0; i < mfccs.length; i++) {
            mfccs[i] = Arrays.copyOfRange(mfccs[i], 0, numCoeffs);
        }
        float[] sumLog = safeLog(sum(powers));
        for (float[] mfcc : mfccs) {
            System.arraycopy(sumLog, 0, mfcc, 0, 1);
        }
        return mfccs;
    }

    /**
     * Calculates mel spectrogram (condensed spectrogram)
     */
    @NotNull
    public float[][] melSpec(@NotNull float[] audio) {
        float[][] spec = powerSpec(audio, audioWindowSize, audioWindowHop, fftSize);
        return safeLog(dot(spec, transpose(filterbanks)));
    }
}
