package cs475;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;

public class Classify {
	static public LinkedList<Option> options = new LinkedList<Option>();
	
	public static void main(String[] args) throws IOException {
		// Parse the command line.
		String[] manditory_args = {"mode"};
		createCommandLineOptions();
		CommandLineUtilities.initCommandLineParameters(args, Classify.options, manditory_args);
	
		String mode = CommandLineUtilities.getOptionValue("mode");
		String data = CommandLineUtilities.getOptionValue("data");
		String predictions_file = CommandLineUtilities.getOptionValue("predictions_file");
		String algorithm = CommandLineUtilities.getOptionValue("algorithm");
		String model_file = CommandLineUtilities.getOptionValue("model_file");
		int sgd_iterations = 20;
		if (CommandLineUtilities.hasArg("sgd_iterations"))
			sgd_iterations = CommandLineUtilities.getOptionValueAsInt("sgd_iterations");
		double sgd_eta0 = 1.0;
		if (CommandLineUtilities.hasArg("sgd_eta0"))
			sgd_eta0 = CommandLineUtilities.getOptionValueAsFloat("sgd_eta0");
		
		double pegasos_lambda = 1e-4; 
		if (CommandLineUtilities.hasArg("pegasos_lambda"))
		    pegasos_lambda = CommandLineUtilities.getOptionValueAsFloat("pegasos_lambda");
		
		if (mode.equalsIgnoreCase("train")) {
			if (data == null || algorithm == null || model_file == null) {
				System.out.println("Train requires the following arguments: data, algorithm, model_file");
				System.exit(0);
			}
			// Load the training data.
			DataReader data_reader = new DataReader(data, true);
			List<Instance> instances = data_reader.readData();
			data_reader.close();
			
			// Train the model.
			if (algorithm.equals("logistic_regression")) {
				Predictor predictor = train(instances, algorithm, sgd_iterations, sgd_eta0);
				saveObject(predictor, model_file);		
			}
			else if (algorithm.equals("pegasos")) {
				Predictor predictor = train(instances, algorithm, sgd_iterations, pegasos_lambda);
				saveObject(predictor, model_file);
			}
			else {
				Predictor predictor = train(instances, algorithm);
				saveObject(predictor, model_file);		
			}		
			
		} else if (mode.equalsIgnoreCase("test")) {
			if (data == null || predictions_file == null || model_file == null) {
				System.out.println("Train requires the following arguments: data, predictions_file, model_file");
				System.exit(0);
			}
			
			// Load the test data.
			DataReader data_reader = new DataReader(data, true);
			List<Instance> instances = data_reader.readData();
			data_reader.close();
			
			// Load the model.
			Predictor predictor = (Predictor)loadObject(model_file);
			evaluateAndSavePredictions(predictor, instances, predictions_file, algorithm);
		} else {
			System.out.println("Requires mode argument.");
		}
	}
	
	private static Predictor train(List<Instance> instances, String algorithm) {
		// TODO Train the model using "algorithm" on "data" and evaluate the model
		ClassificationPredictor predictor = new ClassificationPredictor(algorithm, true);
		predictor.train(instances);
		return predictor; 
	}
	
	private static Predictor train(List<Instance> instances, String algorithm, int iterations, double para) {
		ClassificationPredictor predictor = new ClassificationPredictor(algorithm, iterations, para, true);
		predictor.train(instances);
		return predictor;
	}
	
	private static void evaluateAndSavePredictions(Predictor predictor,
			List<Instance> instances, String predictions_file, String algorithm) throws IOException {
		
		PredictionsWriter writer = new PredictionsWriter(predictions_file);
		predictor = new ClassificationPredictor(predictions_file);
		List<Label> predictedLabels = new ArrayList<Label>();
		Boolean available = true;
		double accuracy = 0.0;
		
		if (algorithm != null && algorithm.equals("logistic_regression")) {
			for (Instance instance : instances) { 
			    LogisticRegressionClassifier classifier = new LogisticRegressionClassifier();
				Label label = classifier.predict(instance); 
				predictedLabels.add(label);
				if (label == null) {
					available = false; 
				}
				writer.writePrediction(label);
			}
			// TODO Evaluate the model if labels are available. 
			if (available) { 
				ClassificationEvaluator evaluator = new ClassificationEvaluator();
				evaluator._logistic_regression = true;
				evaluator._pegasos = false;
				accuracy = evaluator.evaluate(instances, predictor, predictedLabels);
			}
		}
		
		else if (algorithm != null && algorithm.equals("pegasos")) {
			for (Instance instance : instances) { 
			    PegasosClassifier classifier = new PegasosClassifier();
				Label label = classifier.predict(instance); 
				predictedLabels.add(label);
				if (label == null) {
					available = false; 
				}
				writer.writePrediction(label);
			}
			// TODO Evaluate the model if labels are available. 
			if (available) { 
				ClassificationEvaluator evaluator = new ClassificationEvaluator();
				evaluator._pegasos = true;
				evaluator._logistic_regression = false;
				accuracy = evaluator.evaluate(instances, predictor, predictedLabels);
			}
		}
		
		else {
			for (Instance instance : instances) {
				predictor = new ClassificationPredictor(predictions_file);
				Label label = predictor.predict(instance);
				predictedLabels.add(label);
				if (label == null) {
					available = false;
				}
				writer.writePrediction(label);
			}
			// TODO Evaluate the model if labels are available. 
			if (available) {
				ClassificationEvaluator evaluator = new ClassificationEvaluator();
				evaluator._logistic_regression = false;
				evaluator._pegasos = false;
				accuracy = evaluator.evaluate(instances, predictor, predictedLabels);
			}
		}
		writer.close();	
		if (accuracy==0.0) {
			System.out.println("Finish testing, output the preditions to " + predictions_file);
		}
		else {
			System.out.println("The accuracy of "+ predictions_file + " is " + accuracy);
		}
	}

	public static void saveObject(Object object, String file_name) {
		try {
			ObjectOutputStream oos =
				new ObjectOutputStream(new BufferedOutputStream(
						new FileOutputStream(new File(file_name))));
			oos.writeObject(object);
			oos.close();
		}
		catch (IOException e) {
			System.err.println("Exception writing file " + file_name + ": " + e);
		}
	}

	/**
	 * Load a single object from a filename. 
	 * @param file_name
	 * @return
	 */
	public static Object loadObject(String file_name) {
		ObjectInputStream ois;
		try {
			ois = new ObjectInputStream(new BufferedInputStream(new FileInputStream(new File(file_name))));
			Object object = ois.readObject();
			ois.close();
			return object;
		} catch (IOException e) {
			System.err.println("Error loading: " + file_name);
		} catch (ClassNotFoundException e) {
			System.err.println("Error loading: " + file_name);
		}
		return null;
	}
	
	public static void registerOption(String option_name, String arg_name, boolean has_arg, String description) {
		OptionBuilder.withArgName(arg_name);
		OptionBuilder.hasArg(has_arg);
		OptionBuilder.withDescription(description);
		Option option = OptionBuilder.create(option_name);
		
		Classify.options.add(option);		
	}
	
	private static void createCommandLineOptions() {
		registerOption("data", "String", true, "The data to use.");
		registerOption("mode", "String", true, "Operating mode: train or test.");
		registerOption("predictions_file", "String", true, "The predictions file to create.");
		registerOption("algorithm", "String", true, "The name of the algorithm for training.");
		registerOption("model_file", "String", true, "The name of the model file to create/load.");
		registerOption("sgd_eta0", "double", true, "The constant scalar for learning rate in AdaGrad.");
		registerOption("sgd_iterations", "int", true, "The number of SGD iterations.");
		registerOption("pegasos_lambda", "double", true, "The regularization parameter for Pegasos.");
		registerOption("data", "String", true, "The data to use.");
	    	registerOption("gd_eta", "double", true, "The constant scalar for learning rate. default: 0.1");
	    	registerOption("gd_iterations", "int", true, "The number of iterations.default: 20");
		// Other options will be added here.
	}
    
}