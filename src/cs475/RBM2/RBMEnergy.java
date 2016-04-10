package cs475.RBM2;
import cs475.RBM2.*;

import java.util.Random;

public class RBMEnergy {
	
  private RBMParameters parameters;
  private int num_samples;
	
  private double[] x;
  private double[] h;
  private double[] marginal;
	
  private int m;
  private int n;
  private int count;
  private boolean calculated = false;
	
  public RBMEnergy(RBMParameters parameters, int numSamples) {
    this.parameters = parameters;
    this.num_samples = numSamples;
    this.m = parameters.numVisibleNodes();
    this.n = parameters.numHiddenNodes();
    this.count = 0;
    x = new double[m+1];
    h = new double[n+1];
    marginal = new double[n+1];
  }
	
  public double computeMarginal(int j) {	
    if (!calculated) {
      calculateEverything();
    }
    return marginal[j];
  }
		
  void calculateEverything() {
    // calculate all h[.] and x[.] and store marginal probabilities for all j
    initX();
    Random rand = new Random(0);
	  	
    for (int t=1; t<=num_samples; t++) {
      // calculate h[.]
      for (int j=1; j<=n; j++) {
        double xw = 0.0;
        double d = parameters.hiddenBias(j);
        for (int k=1; k<=m; k++) {
          double w = parameters.weight(k, j);
          xw += x[k] * w;
        }
        double p_j = sigmoid(xw + d);
        h[j] = sample(rand, p_j);
        if (h[j] == 1.0) {
          marginal[j] += 1.0;
        }
      }
	  		
      // calculate x[.]
      for (int i=1; i<=m; i++) {
        double hw = 0.0;
        double b = parameters.visibleBias(i);
        for (int k=1; k<=n; k++) {
          double w = parameters.weight(i, k);
          hw += h[k] * w;
        }
        double p_j = sigmoid(hw + b);
        x[i] = sample(rand, p_j);
      }
    }
	  	
    // store marginal probabilities for all j 
    for (int j=1; j<=n; j++) {
      marginal[j] = marginal[j] / (double)num_samples;
    }
	  	
    calculated = true;
  }
	
  private double sample (Random rand, double p) {
    double u = rand.nextDouble();
    double y = 0.0;
    if (u < p) {
      y = 1.0;
    }
    return y;
  }
	
  private void initX() {
    for (int i=2; i<=m; i=i+2) {
      x[i] = 1.0;
    }
  }
	
  private double sigmoid(double z) {
    return 1 / (1 + Math.exp(-z));
  }
	
}