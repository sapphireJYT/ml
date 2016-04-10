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

public class LogisticRegressionClassifier {
  public HashMap<Integer, Double> _w;
  private double _sgd_eta0;
  private int _sgd_iterations;
	
  public LogisticRegressionClassifier (int iterations, double eta0) {
    this._w = new HashMap<Integer, Double>();
    this._sgd_iterations = iterations;
    this._sgd_eta0 = eta0;
  }
	
  public LogisticRegressionClassifier () {
    this._w = new HashMap<Integer, Double>();
  }
	
  public List<Label> train (List<Instance> instances) throws IOException{
    List<Label> labels = new ArrayList<Label>();
    HashMap<Integer, Double> f_sum = new HashMap<Integer, Double>();
    Double I_j = 1.0;
    // Begin the iterations
    for (int i=1; i<=this._sgd_iterations; i++) {
      // Receive X and Y from each instance
      for (Instance instance : instances) {
        int y = Integer.parseInt(instance._label._label);
        HashMap<Integer, Double> x = new HashMap<Integer, Double>(instance._feature_vector._features);
        double wx = computeWX(x);
        Iterator it = x.entrySet().iterator();
        while (it.hasNext()) {
          // Update eta_ij 
          Map.Entry<Integer, Double> entry = (Map.Entry<Integer, Double>) it.next();
          int key = entry.getKey();
					
          double featureValue = entry.getValue();
          double f = y*sign(-wx)*featureValue + (1.0-y)*sign(wx)*(-featureValue);
          if (f_sum.containsKey(key)==false) {
            f_sum.put(key, 0.0);
          }
          f_sum.put(key, f_sum.get(key)+f*f);
				
          // Update w
          if (this._w.containsKey(key)==false) {
            this._w.put(key, 0.0);
          }
          this._w.put(key, this._w.get(key)+(this._sgd_eta0/Math.sqrt(I_j + f_sum.get(key)))*f);
        }
      }	
    }
		
    // Save w for future predictions
    saveW();
    // Calculate labels for training accuracy
    for (Instance instance : instances) {
      int y = sign(computeWX(instance._feature_vector._features))>=0.5 ? 1 : 0;
      labels.add(new ClassificationLabel(y));
    }
    return labels;
  }
	
  private double computeWX (HashMap<Integer, Double> x) {
    double wx = 0.0;
    Iterator it = x.entrySet().iterator();
    while (it.hasNext()) {
      Map.Entry<Integer, Double> entry = (Map.Entry<Integer, Double>) it.next();
      int key = entry.getKey();
      if (this._w.containsKey(key)==false) {
        this._w.put(key, 0.0);
      }
      wx += this._w.get(key)*entry.getValue();
    }
    return wx;
  }
	
  private double sign (double z) {
    return 1/(1+Math.pow(Math.E, -z));
  }
	
  // Save w for future predictions
  private void saveW () throws IOException {
    File file = new File ("logistic_w");
    if (file.exists()) {
      file.delete();
    }
    file = new File ("logistic_w");
    file.createNewFile();
    FileWriter writer = new FileWriter (file, false);
    Iterator it = this._w.entrySet().iterator();
    while (it.hasNext()) {
      Map.Entry<Integer, Double> entry = (Map.Entry<Integer, Double>) it.next();
      writer.write(String.valueOf(entry.getKey())+":"+String.valueOf(entry.getValue()));
      writer.write("\n");
      writer.flush();
    }
    writer.close();
  }
		
  // Predict y during test
  public Label predict (Instance instance) throws FileNotFoundException {
    HashMap<Integer, Double> x = new HashMap<Integer, Double>(instance._feature_vector._features);
    File file = new File("logistic_w");
    DataReader reader = new DataReader(file.getName(), true);
    this._w = reader.readParameters(); 
    int y = sign(computeWX(x))>=0.5 ? 1 : 0;
    Label label = new ClassificationLabel(y);
    return label;
  }

}