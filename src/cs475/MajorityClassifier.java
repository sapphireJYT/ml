package cs475;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

public class MajorityClassifier {
	
  public Label majorityClassify(List<Instance> instances) throws IOException {
    // Store the labels and their corresponding counts in a hash map
    HashMap<String, Integer> labelList = new HashMap<String, Integer>(); 
    for (Instance instance : instances) {
      int count = 1;
      if(instance._label == null) {
        ClassificationPredictor predictor = new ClassificationPredictor(null);
        instance.setLabel(predictor.predict(instance));
      }
      String key = instance._label._label; 
      if (labelList.containsKey(key)) {
        count = labelList.get(key) + 1;
      }
      labelList.put(key, count);
    }
		
    // Find the most common label
    int max = 1;
    Iterator<String> it = labelList.keySet().iterator();
    while (it.hasNext()) {
      String key = it.next();
      max = max<labelList.get(key) ? labelList.get(key) : max;
    }
		
    List<String> commonLabels = new ArrayList<String>();
    it = labelList.keySet().iterator();
    while (it.hasNext()) {
      String key = it.next(); 
      if (labelList.get(key) == max) {
        commonLabels.add(key);
      }
    }
    // Choose randomly among the most common labels 
    int index = new Random().nextInt(commonLabels.size());
    ClassificationLabel label = new ClassificationLabel(Integer.parseInt(commonLabels.get(index)));
		
    // save the label for labeling test data
    saveLabel(label);
    return label;	
  }
	
  private void saveLabel (Label label) throws IOException {
    File file = new File ("majorLabel");
    if (file.exists()) {
      if (file.isDirectory()) {
        file.createNewFile();
      }
    }
    FileWriter writer = new FileWriter (file, false);
    writer.write(label._label);
    writer.flush();
    writer.close();
  }
	
}