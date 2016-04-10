package cs475;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

public class EvenOddClassifier {
	
  public List<Label> evenOddClassify (List<Instance> instances) {
    List<Label> labelList = new ArrayList<Label>();
    for (Instance instance :instances) {
      Label label = evenOddClassify(instance);
      labelList.add(label);
    }
    return labelList;
  }
	
  public Label evenOddClassify (Instance instance) {
    double evenSum = 0.0;
    double oddSum = 0.0;
    HashMap<Integer, Double> features = new HashMap<Integer, Double>(instance._feature_vector._features);
    Iterator<Integer> it = features.keySet().iterator();
		
    while (it.hasNext()) {
      int key = it.next();
      switch (key%2) {
        case 0 : 
        evenSum += features.get(key);
        break;
        case 1 : 
        oddSum += features.get(key);
        break;
      }
    }
    Label label = evenSum >= oddSum ? new ClassificationLabel(1) : new ClassificationLabel(0);	
    return label; 
  }

}