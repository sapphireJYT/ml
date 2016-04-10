package cs475;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class ClassificationEvaluator extends Evaluator implements Serializable {
	
  public Boolean _logistic_regression;
  public Boolean _pegasos;
	
  @Override
  public double evaluate(List<Instance> instances, Predictor predictor){
    double accuracy = 0.0;
    try{
      DataReader reader = new DataReader (predictor._predictions_file, true);
      List<Label> predictedLabels = new ArrayList<Label>(reader.readClassificationLabels());
      int accurateNum = 0;
      if(_logistic_regression == true){ 
        LogisticRegressionClassifier classifier = new LogisticRegressionClassifier();
        for(int i=0; i<predictedLabels.size(); i++){ 
          if(instances.get(i)!=null && instances.get(i)._label==null){
            instances.get(i).setLabel(classifier.predict(instances.get(i)));
            accurateNum--;
          }
          if(instances.get(i)._label._label.equals(predictedLabels.get(i)._label)){
            accurateNum++;
          }
        }
      }
      else if(_pegasos == true){
        PegasosClassifier classifier = new PegasosClassifier();
        for(int i=0; i<predictedLabels.size(); i++){ 
          if(instances.get(i)!=null && instances.get(i)._label==null){
            instances.get(i).setLabel(classifier.predict(instances.get(i)));
            accurateNum--;
          }
          if(instances.get(i)._label._label.equals(predictedLabels.get(i)._label)){
            accurateNum++;
          }
        }
      }
      else{  // majority or even_odd predictor
        for(int i=0; i<predictedLabels.size(); i++){ 
          if(instances.get(i)!=null && instances.get(i)._label==null){
            instances.get(i).setLabel(predictor.predict(instances.get(i)));
            accurateNum--;
          }
          if(instances.get(i)._label._label.equals(predictedLabels.get(i)._label)){
            accurateNum++;
          }
        }	
      }
			
      accuracy = (double)accurateNum/instances.size();
    } catch(FileNotFoundException e){
      e.printStackTrace();
    }	
		
    File file = new File("majorLabel");
    if(file.exists()){
      file.delete();
    }
    return accuracy;
  }
	
  // Evaluate the prediction accuracy of the learning algorithms
  public double evaluate(List<Instance> instances, Predictor predictor, List<Label> predictedLabels){
    double accuracy = 0.0;
    try{
      int accurateNum = 0;
      // Count the correctly predicted labels by the logistic regression classifictaion
      if(_logistic_regression){ 
        LogisticRegressionClassifier classifier = new LogisticRegressionClassifier();
        for(int i=0; i<predictedLabels.size(); i++){ 
          if(instances.get(i)!=null && instances.get(i)._label==null){
            instances.get(i).setLabel(classifier.predict(instances.get(i)));
            accurateNum--;
          }
          if(instances.get(i)._label._label.equals(predictedLabels.get(i).toString())){
            accurateNum++;
          }
        }
      } 
      // Count the correctly predicted labels by the pegasos classifictaion
      else if(_pegasos){ 
        PegasosClassifier classifier = new PegasosClassifier();
        for(int i=0; i<predictedLabels.size(); i++){ 
          if(instances.get(i)!=null && instances.get(i)._label==null){
            instances.get(i).setLabel(classifier.predict(instances.get(i)));
            accurateNum--;
          }
          if(instances.get(i)._label._label.equals(predictedLabels.get(i).toString())){
            accurateNum++;
          }
        }
      } 
      // Count the correctly predicted labels from majority / even_odd algorithms
      else{
        for(int i=0; i<predictedLabels.size(); i++){ 
          if(instances.get(i)!=null && instances.get(i)._label==null){
            instances.get(i).setLabel(predictor.predict(instances.get(i)));
            accurateNum--;
          }
          if(instances.get(i)._label._label.equals(predictedLabels.get(i).toString())){
            accurateNum++;
          }
        }	
      }
			
      accuracy = (double)accurateNum/predictedLabels.size();
    } catch(FileNotFoundException e){
      e.printStackTrace();
    }	
		
    File file = new File ("majorLabel");
    if(file.exists()){
      file.delete();
    }
    return accuracy;
  }
	
  public double evaluateTraining (List<Instance> instances, Label label){
    int accurateNum = 0; 
    for(Instance instance : instances){
      if(instance!=null && instance._label==null){
        ClassificationPredictor predictor = new ClassificationPredictor(null);
        instance.setLabel(predictor.predict(instance));
        accurateNum--;
      }
      if(instance._label._label.equals(label._label)){
        accurateNum++;
      }
    }
    double accuracy = (double)accurateNum/instances.size();
    return accuracy;
  }
	
  public double evaluateTraining (List<Instance> instances, List<Label> labelList){
    int accurateNum = 0;
    for(int i=0; i<labelList.size(); i++){
      if(instances.get(i)!=null && instances.get(i)._label==null){
        ClassificationPredictor predictor = new ClassificationPredictor(null);
        instances.get(i).setLabel(predictor.predict(instances.get(i)));
        accurateNum--;
      }
      if(instances.get(i)._label._label.equals(labelList.get(i)._label)){
        accurateNum++; 
      } 
    }
    double accuracy = (double)accurateNum/instances.size();
    return accuracy; 
  }
	
}