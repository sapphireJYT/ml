package cs475.RBM;

public class RBMEnergy {
	
  private RBMParameters _parameters;
  private int _iters;
  private double _eta;
	
  private int m; 
  private int n;
  private int T;
  private double Z;
  private double[][] x;
  private double[][] h;
	
  public RBMEnergy(RBMParameters parameters, int iters, double eta) {
    this._parameters = parameters;
    this._iters = iters;
    this._eta = eta; 
  }
	
  public void learning() {
    // Initialize W, b, d
    m = this._parameters.numVisibleNodes();
    n = this._parameters.numHiddenNodes();
    T = this._parameters.numExamples();
		
    initParameters();
		
    for (int k=0; k<_iters; k++) {
      // E Step
      Z = computeZ();
      // M step
      updateParameters();
    }
  }
	
  private void updateParameters() {
    double[][] w = new double[m][n];
    double[] b = new double[m];
    double[] d = new double[n];
    w = updateW();
    b = updateB();
    d = updateD();
		
    for (int i=0; i<m; i++) {
      _parameters.setVisibleBias(i, b[i]);
      for (int j=0; j<n; j++) {
        _parameters.setWeight(i, j, w[i][j]);
        _parameters.setHiddenBias(j, d[j]);
      }
    }
  }
	
  private double[][] updateW () {
    double[][] w = new double[m][n];
    for (int i=0; i<m; i++) {
      for (int j=0; j<n; j++) {
        double delta = 0.0;
        for (int t=0; t<T; t++) {
          double sigmoid = sigmoid(dotproduct(t,j) + _parameters.getHiddenBias(j));
          delta += sigmoid * _parameters.getExample(t, i);
        }
        double p = 0.0;
        for (int a=0; a<Math.pow(2,m); a++) {
          for (int b=0; b<Math.pow(2,n); b++) {
            double x_i = x[a][i];
            double h_j = h[b][j];
						
            if (x_i==1 && h_j==1) {
              double E = computeE(a, b);
              p += Math.exp(-E);
            } 
          }
        }
        p = p/Z; 
        delta += -T*p;
        w[i][j] = _parameters.getWeight(i, j) + _eta*delta;  
      }
    }
    return w;
  }
	
  private double[] updateB () {
    double[] B = new double[m];
    for (int i=0; i<m; i++) {
      double delta = 0.0;
      for (int t=0; t<T; t++) {
        delta += _parameters.getExample(t, i);
      }
      double p = 0.0;
      for (int a=0; a<Math.pow(2,m); a++) {
        for (int b=0; b<Math.pow(2,n); b++) {
          double x_i = x[a][i];
          if (x_i==1) {
            double E = computeE(a, b);
            p += Math.exp(-E);	
          }
        }
      }
      p = p/Z; 
      delta += -T*p; 
      B[i] = _parameters.getVisibleBias(i) + _eta*delta; 
    }
    return B;
  }
	
  private double[] updateD () {
    double[] d = new double[n];
    for (int j=0; j<n; j++) {
      double delta = 0.0;
      for (int t=0; t<T; t++) {
        double sigmoid = sigmoid(dotproduct(t,j) + _parameters.getHiddenBias(j));
        delta += sigmoid;
      }
      double p = 0.0;
      for (int a=0; a<Math.pow(2,m); a++) {
        for (int b=0; b<Math.pow(2,n); b++) {
          double h_j = h[b][j];
					
          if (h_j==1) {
            double E = computeE(a, b);
            p += Math.exp(-E); 
          }	
        }
      }
      p = p/Z; 
      delta += -T*p; 
      d[j] = _parameters.getHiddenBias(j) + _eta*delta; 
    }  
    return d;
  }

  private double computeE(int a, int b) {
    double E = 0.0;
    double[] xw = new double[n];
    for (int jj=0; jj<n; jj++) {
      xw[jj] = 0.0;
      for (int ii=0; ii<m; ii++) {
        double x_ii = x[a][ii];
        double w_ij = _parameters.getWeight(ii, jj);
        xw[jj] += x_ii*w_ij;
      }
    }
    for (int jj=0; jj<n; jj++) {
      double h_jj = h[b][jj];
      E += xw[jj]*h_jj;
    }
    for (int ii=0; ii<m; ii++) {
      double x_ii = x[a][ii];
      double b_ii = _parameters.getVisibleBias(ii); 
      E += b_ii*x_ii;
    }
    for (int jj=0; jj<n; jj++) {
      double h_jj = h[b][jj];
      double d_jj = _parameters.getHiddenBias(jj);  
      E += d_jj*h_jj;
    }
    return -E;
  }
	
  private double dotproduct (int t, int j) {
    double xw = 0.0;
    for (int i=0; i<m; i++) {
      double x = this._parameters.getExample(t, i);
      double w = this._parameters.getWeight(i, j);
      xw += x*w;
    }
    return xw;
  }
	
  private double sigmoid (double z) {	
    return 1 / (1 + Math.exp(-z));
  }
	
  private double computeZ () {
    double Z = 0.0;	
    for (int a=0; a<Math.pow(2,m); a++) {
      for (int b=0; b<Math.pow(2,n); b++) {
        double E = computeE(a, b);
        Z += Math.exp(-E);
      }
    }
    return Z;
  }
	
  private void initParameters() {
    // Consider all the possibilities of x
    x = new double[(int)Math.pow(2,m)][m];
    for (int k=0; k<Math.pow(2,m); k++) {
      String str = Integer.toBinaryString(k);
      for (int l=0; l<str.length(); l++) {
        x[k][m-l-1] = Character.getNumericValue(str.charAt(str.length()-l-1)); 
      }
      for (int a=0; a<m-str.length(); a++) {
        x[k][a] = 0.0;
      }	
    }
		
    // Consider all the possibilities of h
    h = new double[(int)Math.pow(2,n)][n];
    for (int k=0; k<Math.pow(2,n); k++) {
      String str = Integer.toBinaryString(k);
      for (int l=0; l<str.length(); l++) {
        h[k][n-l-1] = Character.getNumericValue(str.charAt(str.length()-l-1)); 
      }
      for (int a=0; a<n-str.length(); a++) { 
        h[k][a] = 0.0;
      }	
    }
		
    for (int i=0; i<m; i++) {
      for (int j=0; j<n; j++) {
        if (i%2==1) {
          _parameters.setVisibleBias(i, 1);
        }	
        if(j%2==1){
          _parameters.setWeight(i, j, 1);
          _parameters.setHiddenBias(j, 1);
        }
      }
    }
  }
	
  public void printParameters() {
    //NOTE: Do not modify this function
    for (int i=0; i<_parameters.numVisibleNodes(); i++)
      System.out.println("b_" + i + "=" + _parameters.getVisibleBias(i));
    for (int i=0; i<_parameters.numHiddenNodes(); i++)
      System.out.println("d_" + i + "=" + _parameters.getHiddenBias(i));
    for (int i=0; i<_parameters.numVisibleNodes(); i++)
      for (int j=0; j<_parameters.numHiddenNodes(); j++)
        System.out.println("W_" + i + "_" + j + "=" + _parameters.getWeight(i,j));
  }
    
}