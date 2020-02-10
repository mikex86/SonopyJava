# Sonopy Java Implementation
![Java CI](https://github.com/GommeAntiLegit/SonopyJava/workflows/Java%20CI/badge.svg?branch=master&event=push)

This is a Java implementation of the Audio feature extraction library
[Sonopy by MycroftAI](https://github.com/MycroftAI/sonopy).

It is a lightweight library used to calculate the MFCCs of an audio signal. 
It implements the following audio vectorization functions:

- Power spectrogram
- Mel spectrogram
- Mel frequency cepstrum coefficient spectrogram

## Features
- Lightweight
- Tiny, readable source code
- Visualize steps in calculation

Read more about the library on https://github.com/MycroftAI/sonopy.

## Usage

```java
import me.gommeantilegit.sonopy.Sonopy;

float[] audio = {...};

float[][] powers = Sonopy.powerSpec(audio, audioWindowSize, audioWindowHop, fftSize); // static function

// Functions that depend on filterbanks values, are instance bound to avoid recalculation with same parameters
Sonopy sonopy = new Sonopy(sampleRate, audioWindowSize, audioWindowHop, fttSize, numFilters);
float[][] mels = sonopy.melSpec(audio);

float[][] mfccs = sonopy.mfccSpec(audio, numCoeffs);
float[][] filters = Sonopy.filterbanks(sampleRate, numFilters, fftLen); // Probably not ever useful

// return_parts parameter does not exist in Sonopy.mfccSpec(...) due to Java language limitations
```

## Installation

Add the add the latest release jar to your classpath using your buildsystem or IDE.

If there are any problems, try adding the [Jetbrains Annotations](https://mvnrepository.com/artifact/org.jetbrains/annotations) library.

## Building from Source

```
gradlew build
```

The jar will be placed in `./build/libs/`

## Credits
- [Sonopy by MycroftAI](https://github.com/MycroftAI/sonopy)
- [Fast Fourier Transform by Danny Su and Hanns Holger Rutz](https://github.com/Sciss/SpeechRecognitionHMM/blob/master/src/main/java/org/ioe/tprsa/audio/feature/FFT.java)
