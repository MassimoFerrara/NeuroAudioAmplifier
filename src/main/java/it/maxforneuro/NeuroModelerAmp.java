package it.maxforneuro;

import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import javax.sound.sampled.AudioFormat;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class NeuroModelerAmp {
    public static MultiLayerNetwork buildRNN(int numInput, int numHidden, int numOutputs, double learningRate) {
        return new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam.Builder().learningRate(learningRate).build())
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .list()
                .layer(0, new LSTM.Builder()
                        .nIn(numInput)
                        .nOut(numHidden)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(numHidden)
                        .nOut(numOutputs)
                        .activation(Activation.IDENTITY)
                        .build())
                .build());
    }

    public static List<DataSet> generateTrainingDataForRNN(double[] cleanAudioSamples, double[] distortedAudioSamples) throws IOException {
        List<DataSet> trainingData = new ArrayList<>();

        int minLength = Math.min(cleanAudioSamples.length, distortedAudioSamples.length);
        minLength = 1000; // Unfortunately, I have to limit to avoid extremely slow processing

        // Create training data
        for (int i = 0; i < minLength; i++) {
            // Add a time dimension to the input for RNN
            INDArray input = Nd4j.create(new double[]{cleanAudioSamples[i]});
            INDArray output = Nd4j.create(new double[]{distortedAudioSamples[i]});

            // Add the time dimension
            input = input.reshape(1, 1, 1);

            // Ensure the shape of the output reflects the shape of your labels
            output = output.reshape(1, 1); // For example, if labels are 1D arrays

            trainingData.add(new DataSet(input, output));
        }

        return trainingData;
    }

    public static MultiLayerNetwork loadNeuralNetworkModel(String ampFileModeler) throws IOException {
        InputStream inputStream = new BufferedInputStream(new FileInputStream(ampFileModeler));
        return ModelSerializer.restoreMultiLayerNetwork(inputStream);
    }

    public static double[] amplifyByModel(MultiLayerNetwork neuralNetwork, double[] cleanSignalToApplyAmp) {
        // Note: In this implementation, amplification is achieved by simply taking the model predictions for each sample
        double[] amplifiedSignal = new double[cleanSignalToApplyAmp.length];

        for (int i = 0; i < cleanSignalToApplyAmp.length; i++) {
            // Create input for the model
            INDArray input = Nd4j.create(new double[]{cleanSignalToApplyAmp[i]});
            input = input.reshape(1, 1, 1);

            // Get prediction from the model
            INDArray output = neuralNetwork.output(input);

            // Add the amplified prediction to the resulting array
            amplifiedSignal[i] = output.getDouble(0);
        }

        return amplifiedSignal;
    }

    public static void createModel(String cleanSignalFileName, String distortedSignalFileName) throws Exception {
        Pair<double[], AudioFormat> cleanSignalData = WavFileProcessor.readWavFile(cleanSignalFileName);
        Pair<double[], AudioFormat> distortedSignalData = WavFileProcessor.readWavFile(distortedSignalFileName);
        double[] cleanSignal = cleanSignalData.getLeft();
        double[] distortedSignal = distortedSignalData.getLeft();

        // Neural network configuration (using the RNN model as an example)
        int numInput = 1; // Number of inputs (e.g., audio samples)
        int numOutputs = 1; // Number of outputs (e.g., predicted value)
        int numHidden = 64; // Number of neurons in the hidden layer
        double learningRate = 0.001; // Learning rate

        MultiLayerNetwork network = NeuroModelerAmp.buildRNN(numInput, numHidden, numOutputs, learningRate);
        network.init();

        // Configure your training data
        List<DataSet> trainingData = NeuroModelerAmp.generateTrainingDataForRNN(cleanSignal, distortedSignal);

        // Train the neural network
        int numEpochs = 10;
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            for (DataSet dataSet : trainingData) {
                network.fit(dataSet);
            }
        }

        ModelSerializer.writeModel(network, new File("amplifier_model.zip"), true);
    }

    public static void main(String[] args) throws Exception {
        String cleanSignalFileName = "clean-uno.wav";
        String distortedSignalFileName = "distorted-uno.wav";
        createModel(cleanSignalFileName, distortedSignalFileName);
    }
}
