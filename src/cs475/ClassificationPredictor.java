package cs475;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class ClassificationPredictor extends Predictor implements Serializable{

	public Boolean _train;
	public String _algorithm;
	public int _sgd_iterations;
	public double _sgd_eta0; 
	public double _pegasos_lambda;
	
	public ClassificationPredictor (String algorithm, Boolean train) {
		this._algorithm = algorithm;
	}
	
	public ClassificationPredictor (String algorithm, int iterations, double para, Boolean train) {
		this._algorithm = algorithm;
		this._sgd_iterations = iterations;
		if (algorithm.equals("logistic_regression")) {
			this._sgd_eta0 = para;
		}
		else if (algorithm.equals("pegasos")) {
			this._pegasos_lambda = para;
		}	
	}
	
	public ClassificationPredictor (String predictionsFileName) {
		this._predictions_file = predictionsFileName; 
	}
	
	@Override
	public void train(List<Instance> instances) {
		double trainingAccuracy = 0.0;
		ClassificationEvaluator evaluator = new ClassificationEvaluator();
		
		// Majority Classification
		if (_algorithm.equals("majority")) {
			MajorityClassifier classifier = new MajorityClassifier();
			try {
				Label label = classifier.majorityClassify(instances);
				// Evaluate the training accuracy
				trainingAccuracy = evaluator.evaluateTraining(instances, label);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		
		// Even_Odd Classification
		else if (_algorithm.equals("even_odd")) {
			EvenOddClassifier classifier = new EvenOddClassifier();
			List<Label> labelList = new ArrayList<Label>();
			labelList = classifier.evenOddClassify(instances);
			// Evaluate the training accuracy
			trainingAccuracy = evaluator.evaluateTraining(instances, labelList);
		}	
		
		else if (_algorithm.equals("logistic_regression")) {
			LogisticRegressionClassifier classifier = new LogisticRegressionClassifier(_sgd_iterations, _sgd_eta0);
			List<Label> labelList = new ArrayList<Label>();
			try {
				labelList = classifier.train(instances);
			} catch (IOException e) {
				e.printStackTrace();
			}
			// Evaluate the training accuracy
			trainingAccuracy = evaluator.evaluateTraining(instances, labelList);
		}
		
		else if (_algorithm.equals("pegasos")) {
			PegasosClassifier classifier = new PegasosClassifier(_sgd_iterations, _pegasos_lambda);
			List<Label> labelList = new ArrayList<Label>();
			try {
				labelList = classifier.train(instances);
			} catch (IOException e) {
				e.printStackTrace();
			}
			trainingAccuracy = evaluator.evaluateTraining(instances, labelList);
		}
		
		System.out.println("The training accuracy of " + this._algorithm + " algorithm is " + trainingAccuracy + ".");
	}

	public Label predict(Instance instance) {
		Label label = null;
		File file = new File("majorLabel");
		if (file.exists()) {
		// get the most common label obtained by precious training
			try {
				DataReader reader = new DataReader(file.getName(), true);
				label = reader.readMajorLabel(); 
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
			this._algorithm = "";
		}
		
		else {
		// predict the label;	
			EvenOddClassifier classifier = new EvenOddClassifier();
			label = classifier.evenOddClassify(instance);
		}
		
		return label;
	}
	
}