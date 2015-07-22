package cs475;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Scanner;


public class DataReader {

	private Scanner _scanner;
	// Classification or regression?
	private boolean _classification;

	public DataReader(String filename, boolean classification) throws FileNotFoundException {
		this._scanner = new Scanner(new BufferedInputStream(new FileInputStream(filename)));
		this._classification = classification;
	}
	
	public void close() {
		this._scanner.close();
	}
	
	public List<Instance> readData() {
		ArrayList<Instance> instances = new ArrayList<Instance>();
		
		while (this._scanner.hasNextLine()) {
			String line = this._scanner.nextLine();
			if (line.trim().length() == 0)
				   continue;
			
			FeatureVector feature_vector = new FeatureVector();
		
			// Divide the line into features and label.
			String[] split_line = line.split(" ");
			String label_string = split_line[0];
			Label label = null;
			if (this._classification) { 
				int int_label = Integer.parseInt(label_string);
				if (int_label != -1) {
					label = new ClassificationLabel(int_label);
				}
			} else {
				try {
					double double_label= Double.parseDouble(label_string);
					label = new RegressionLabel(double_label);
				} catch (Exception e) {
					
				}
			}
			for (int ii = 1; ii < split_line.length; ii++) {
				String item = split_line[ii];
				String name = item.split(":")[0];
				int index = Integer.parseInt(name);
				double value = Double.parseDouble(item.split(":")[1]);
				
				if (value != 0)
					feature_vector.add(index, value); 
			}
			
			Instance instance = new Instance(feature_vector, label);
			instances.add(instance); 
		}		
		return instances;
	}
	
	public Label readMajorLabel () {
		Label label = null;
		if (this._scanner.hasNext()) {
			label = new ClassificationLabel(Integer.parseInt(this._scanner.next()));
		}
		return label;
	}
	
	public HashMap<Integer, Double> readParameters () {
		HashMap<Integer, Double> parameters = new HashMap<Integer, Double>();
		while (this._scanner.hasNextLine()) {
			String line = this._scanner.nextLine();
			if (line.trim().length() == 0)
				   continue;
			parameters.put(Integer.parseInt(line.split(":")[0]), Double.parseDouble(line.split(":")[1]));
		}
		return parameters;
	}
	
	public List<Label> readClassificationLabels () throws FileNotFoundException {
		List<Label> classificationLabels = new ArrayList<Label>();
		while (this._scanner.hasNextLine()) {
			String line = this._scanner.nextLine();
			int y;
			if (line.trim().length() == 0)
				   y = -1;
			y = Integer.parseInt(line.trim());
			Label label = new ClassificationLabel(y);
			classificationLabels.add(label);
		}
		return classificationLabels;
	}
	
}