/**
* 
*/
package cs475;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
* sub-GrAdient SOlver for SVM
* @author yatbear
*
*/
public class PegasosClassifier {
	
  private int sgd_iterations;
  private double pegasos_lambda;
  HashMap<Integer, Double> w;
	
  public PegasosClassifier() {
    this.w = new HashMap<Integer, Double>();
  }
	
  public PegasosClassifier(int iterations, double lambda) {
    this.sgd_iterations = iterations;
    this.pegasos_lambda = lambda;
    this.w = new HashMap<Integer, Double>();
  }
	
	
  public List<Label> train(List<Instance> instances) throws IOException {
    // Initialize w 
    initW (instances);
    int t = 1; // time step
    // For each iteration
    for (int i=0; i<this.sgd_iterations; i++) {
      // Go through all the instances
      for (Instance instance : instances) { 
        int y = Integer.parseInt(instance._label._label);
        // Convert label 0 to label -1
        y = y==0? -1 : 1;
        HashMap<Integer, Double> x = new HashMap<Integer, Double> (instance._feature_vector._features);
				
        // Update w at each time step t
        double indicator = y*computeWX(x)<1 ? 1.0 : 0; 
        Iterator<Integer> wIt = w.keySet().iterator();
        while (wIt.hasNext()) {
          int k = wIt.next();
          double x_i = 0.0;
          if (x.containsKey(k)) {
            x_i = x.get(k);
          }
          w.put(k, (1.0-1.0/t)*w.get(k) + (1.0/(pegasos_lambda*t))*indicator*y*x_i);	
        }		
        t++;
      }
    }
		
    // Save w to files for future predictions
    saveW();
    // Use the trained w to predict labels for training accuracy evaluation
    List<Label> labels = new ArrayList<Label>();
    for (Instance instance : instances) {
      int y = computeWX(instance._feature_vector._features)>=0 ? 1 : 0;
      Label label = new ClassificationLabel(y);
      labels.add(label);
    }	
    return labels;
  }
	
  // Predict the label of an instance given its feature 
  public Label predict(Instance instance) throws FileNotFoundException {
    HashMap<Integer, Double> x = new HashMap<Integer, Double>(instance._feature_vector._features); 
    File file = new File("pegasos_w");
    DataReader reader = new DataReader(file.getName(), true);
    this.w = reader.readParameters();
    int y = computeWX(x)>=0 ? 1 : 0; 
    Label label = new ClassificationLabel(y);
    return label;
  }

  // Initialize the weigh vector to be zero
  private void initW(List<Instance> instances) {
    int max = 0;
    // Find the dimension of w
    for (Instance instance : instances) {
      HashMap<Integer, Double> feature = new HashMap<Integer, Double> (instance._feature_vector._features);
      Iterator<Integer> it = feature.keySet().iterator();
      while (it.hasNext()) {
        int key = it.next();
        if (max<key) {
          max = key;
        }
      }
    }
    for (int i=1; i<=max; i++) {
      w.put(i, 0.0);
    }
  }
	
  // Compute the dot productor of weight vector w and feature vector x
  private double computeWX(HashMap<Integer, Double> x) {
    double wx = 0.0;
    Iterator it = x.entrySet().iterator();
    while (it.hasNext()) {
      Map.Entry<Integer, Double> entry = (Map.Entry<Integer, Double>) it.next();
      if (w.containsKey(entry.getKey()))
        wx += this.w.get(entry.getKey())*entry.getValue();
      else
        w.put(entry.getKey(), 0.0);
    }
    return wx;
  }	
	

  // Save w for future predictions
  private void saveW() throws IOException {
    File file = new File ("pegasos_w");
    if (file.exists()) {
      file.delete();
    }
    file = new File ("pegasos_w");
    file.createNewFile();
    FileWriter writer = new FileWriter (file, false);
    Iterator it = this.w.entrySet().iterator();
    while (it.hasNext()) {
      Map.Entry<Integer, Double> entry = (Map.Entry<Integer, Double>) it.next();
      writer.write(String.valueOf(entry.getKey())+":"+String.valueOf(entry.getValue()));
      writer.write("\n");
      writer.flush();
    }
    writer.close();
  }
	
}