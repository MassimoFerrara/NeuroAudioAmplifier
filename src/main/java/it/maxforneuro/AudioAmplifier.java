package it.maxforneuro;

import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import javax.sound.sampled.AudioFormat;

public class AudioAmplifier {

    public static void main(String[] args) throws Exception {
        MultiLayerNetwork neuralNetwork = NeuroModelerAmp.loadNeuralNetworkModel("amplifier_model.zip");
        Pair<double[],AudioFormat> cleanAudioSamples = WavFileProcessor.readWavFile("clean-guitar-lick-81169-24.wav");
        double[] cleanAudioApplyAmpSamples = NeuroModelerAmp.amplifyByModel(neuralNetwork,cleanAudioSamples.getLeft());
        WavFileProcessor.writeWavFile(cleanAudioApplyAmpSamples, cleanAudioSamples.getRight(), "clean-guitar-lick-81169-AMP-APPLY.wav");
    }
}
