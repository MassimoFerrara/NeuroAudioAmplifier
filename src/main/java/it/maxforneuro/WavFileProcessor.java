package it.maxforneuro;

import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;

import javax.sound.sampled.*;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

public class WavFileProcessor {

    public static Pair<double[],AudioFormat> readWavFile(String filePath) throws Exception {

        File cleanSignalFile = getFileFromResources(filePath);
        AudioInputStream audioInputStream = null;

        audioInputStream = AudioSystem.getAudioInputStream(cleanSignalFile);

        AudioFormat format = audioInputStream.getFormat();
        AudioFormat.Encoding encoding = format.getEncoding();

        System.out.println("Audio Format: " + format);
        System.out.println("Sample Rate: " + format.getSampleRate());
        System.out.println("Bit Depth: " + format.getSampleSizeInBits());
        System.out.println("Channels: " + format.getChannels());
        System.out.println("Encoding Type: " + encoding.toString());

        return new ImmutablePair(convertAudioInputStreamToFloatArray(audioInputStream),format);

    }

    public static void writeWavFile(double[] amplifiedSignal, AudioFormat format, String outputFilePath) throws IOException {
        File outputFile = new File(outputFilePath);
        AudioInputStream audioInputStream2 = new AudioInputStream(
                new ByteArrayInputStream(convertFloatArrayToByteArray24bit(amplifiedSignal)),
                format,
                amplifiedSignal.length
        );
        System.out.println("Successfully wrote audio file.");

        AudioFormat outputFormat = new AudioFormat(
                AudioFormat.Encoding.PCM_SIGNED,
                format.getSampleRate(),
                24,
                format.getChannels(),
                format.getChannels() * 3,
                format.getSampleRate(),
                false
        );

        AudioInputStream convertedInputStream = new AudioInputStream(
                audioInputStream2,
                outputFormat,
                amplifiedSignal.length
        );

        AudioSystem.write(convertedInputStream, AudioFileFormat.Type.WAVE, outputFile);

        System.out.println("Successfully read and wrote audio file.");
    }

    private static byte[] convertFloatArrayToByteArray24bit(double[] floatArray) {
        byte[] byteArray = new byte[floatArray.length * 3];

        ByteBuffer buffer = ByteBuffer.wrap(byteArray);
        buffer.order(ByteOrder.LITTLE_ENDIAN);

        for (int i = 0; i < floatArray.length; i++) {
            int sample = (int) (floatArray[i] * 8388607.0);
            buffer.put((byte) (sample & 0xFF));
            buffer.put((byte) ((sample >> 8) & 0xFF));
            buffer.put((byte) ((sample >> 16) & 0xFF));
        }

        return byteArray;
    }

    private static File getFileFromResources(String fileName) {
        File file;
        try {
            URL resourceUrl = Objects.requireNonNull(WavFileProcessor.class.getClassLoader().getResource(fileName));
            file = new File(resourceUrl.toURI());
        } catch (URISyntaxException | NullPointerException e) {
            throw new RuntimeException("File not found: " + fileName, e);
        }

        return file;
    }

    private static double[] convertAudioInputStreamToFloatArray(AudioInputStream audioInputStream) throws IOException {
        byte[] audioBytes = new byte[(int) (audioInputStream.getFrameLength() * audioInputStream.getFormat().getFrameSize())];
        audioInputStream.read(audioBytes);

        return convertByteArrayToFloatArray24bit(audioBytes);
    }

    private static double[] convertByteArrayToFloatArray24bit(byte[] audioBytes) {
        double[] floatArray = new double[audioBytes.length / 3];

        for (int i = 0, j = 0; i < audioBytes.length; i += 3, j++) {
            int sample = ((audioBytes[i] & 0xFF) | ((audioBytes[i + 1] & 0xFF) << 8) | ((audioBytes[i + 2] & 0xFF) << 16));
            floatArray[j] = sample / 8388608.0f;
        }

        return floatArray;
    }

}
