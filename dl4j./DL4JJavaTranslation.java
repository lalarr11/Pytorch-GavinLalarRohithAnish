import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import java.util.HashMap;
import java.util.Map;

public class NMT_DL4J {
    
    // === STEP 1: DATA PREPROCESSING ===
    public static Map<String, Integer> buildVocab(String[] sentences) {
        Map<String, Integer> vocab = new HashMap<>();
        vocab.put("<unk>", 0);
        vocab.put("<pad>", 1);
        vocab.put("<sos>", 2);
        vocab.put("<eos>", 3);
        int index = 4;
        for (String sentence : sentences) {
            for (String word : sentence.split(" ")) {
                vocab.putIfAbsent(word, index++);
            }
        }
        return vocab;
    }

    public static int[] sentenceToTensor(String sentence, Map<String, Integer> vocab) {
        String[] words = sentence.split(" ");
        int[] tensor = new int[words.length];
        for (int i = 0; i < words.length; i++) {
            tensor[i] = vocab.getOrDefault(words[i], vocab.get("<unk>"));
        }
        return tensor;
    }

    // === STEP 2: BUILD THE ENCODER MODEL ===
    public static MultiLayerNetwork buildEncoder(int inputDim, int hiddenDim) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new LSTM.Builder().nIn(inputDim).nOut(hiddenDim)
                        .activation(Activation.TANH)
                        .build())
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

    // === STEP 3: BUILD THE DECODER MODEL ===
    public static MultiLayerNetwork buildDecoder(int outputDim, int hiddenDim) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new LSTM.Builder().nIn(hiddenDim).nOut(hiddenDim)
                        .activation(Activation.TANH)
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(hiddenDim).nOut(outputDim)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        return model;
    }

    // === STEP 4: TRANSLATION FUNCTION ===
    public static String translateSentence(String sentence, MultiLayerNetwork encoder, MultiLayerNetwork decoder,
                                           Map<String, Integer> sourceVocab, Map<String, Integer> targetVocab) {
        int[] inputTensor = sentenceToTensor(sentence, sourceVocab);
        int targetIdx = targetVocab.get("<sos>");
        StringBuilder outputSentence = new StringBuilder();

        for (int i = 0; i < 10; i++) {
            targetIdx = (int) decoder.output(new int[]{targetIdx}).getDouble(0);
            if (targetIdx == targetVocab.get("<eos>")) break;
            outputSentence.append(targetIdx).append(" ");
        }
        return outputSentence.toString();
    }

    // === MAIN FUNCTION TO RUN EVERYTHING ===
    public static void main(String[] args) {
        // Sample data
        String[] sourceSentences = {"hello how are you", "I love machine learning"};
        String[] targetSentences = {"hola cómo estás", "me encanta el aprendizaje automático"};

        // Build vocab
        Map<String, Integer> sourceVocab = buildVocab(sourceSentences);
        Map<String, Integer> targetVocab = buildVocab(targetSentences);

        // Create models
        MultiLayerNetwork encoder = buildEncoder(sourceVocab.size(), 256);
        MultiLayerNetwork decoder = buildDecoder(targetVocab.size(), 256);

        // Example translation
        String translatedText = translateSentence("hello how are you", encoder, decoder, sourceVocab, targetVocab);
        System.out.println("Translated Sentence: " + translatedText);
    }
}
