
package cs475.loopMRF;

import java.util.Hashtable;

public class LoopyBP {

  private LoopMRFPotentials potentials;
  private int iterations;
  private int n;
  private int k;
  private double[][][] mu_fx;
  private double[][][] mu_xf;

  public LoopyBP(LoopMRFPotentials p, int iterations) {
    this.potentials = p;
    this.iterations = iterations;
  }
	
  private void compute_mu_forward () {
    for (int i=1; i<=n; i++) {
      // Compute mu from f to x
      int fkey = n + i;
      int xkey = (i+1)>n ? (i+1)%n : i+1;    
      for (int a=1; a<=k; a++) {
        mu_fx[fkey][xkey][a] = 0.0;
        for (int b=1; b<=k; b++) {  // x is k-ary
          double f = potentials.potential(n+i, b, a); 
          int xkey0 = xkey==1 ? n : xkey-1;
          mu_fx[fkey][xkey][a] += mu_xf[xkey0][fkey][b] * f;
        }
        /*
          System.out.print("mu_fx");
          System.out.print(" f_"+String.valueOf(fkey)+" x_"+String.valueOf(xkey)+" "+String.valueOf(a)+" : ");
          System.out.println(mu_fx[fkey][xkey][a]);
          */
      }
			
      // Compute mu from x to f
      xkey = 1 + i % n;
      fkey = n + 1 + i % n;
      for (int a=1; a<=k; a++) {
        int fkey0 = fkey==n+1 ? 2*n : fkey-1;
        mu_xf[xkey][fkey][a] = mu_fx[xkey][xkey][a] * mu_fx[fkey0][xkey][a];
        /*
          System.out.print("mu_xf");
          System.out.print(" x_"+String.valueOf(xkey)+" f_"+String.valueOf(fkey)+" "+String.valueOf(a)+" : ");
          System.out.println(mu_xf[xkey][fkey][a]);
          */
      }	 
    }
  }
	
  private void compute_mu_backward () {
    for (int i=n; i>0; i--) {
      // Compute mu from f to x
      int fkey = n + i;
      int xkey = i;    
      for (int a=1; a<=k; a++) { 
        mu_fx[fkey][xkey][a] = 0.0;
        for (int b=1; b<=k; b++) {  // x is k-ary
          double f = potentials.potential(n+i, a, b); 
          int xkey1 = xkey==n ? 1 : xkey+1; 
          mu_fx[fkey][xkey][a] += mu_xf[xkey1][fkey][b] * f;
        }
        /*
          System.out.print("mu_fx");
          System.out.print(" f_"+String.valueOf(fkey)+" x_"+String.valueOf(xkey)+" "+String.valueOf(a)+" : ");
          System.out.println(mu_fx[fkey][xkey][a]);
          */
      }
			
      // Compute mu from x to f
      xkey = i;
      fkey = i==1 ? 2*n : n+(i-2)%n+1;
      for (int a=1; a<=k; a++) {
        int fkey1 = fkey==2*n ? n+1 : fkey+1;
        mu_xf[xkey][fkey][a] = mu_fx[xkey][xkey][a] * mu_fx[fkey1][xkey][a];
        /*
          System.out.print("mu_xf");
          System.out.print(" x_"+String.valueOf(xkey)+" f_"+String.valueOf(fkey)+" "+String.valueOf(a)+" : ");
          System.out.println(mu_xf[xkey][fkey][a]);
          */
      }	 
    }
  }
	
  public double[] marginalProbability(int x_i) {
    n = potentials.loopLength();
    k = potentials.numXValues();
    double[] marginals = new double[k+1];
	
    mu_fx = new double[2*n+1][n+1][k+1];  // mu_fx[fkey][xkey][k]
    mu_xf = new double[n+1][2*n+1][k+1];  // mu_xf[xkey][fkey][k]
		
    for (int i=1; i<=k; i++) {
      mu_xf[1][n+1][i] = 1.0;
      mu_xf[1][2*n][i] = 1.0;
    }
		
    // Store messages from f_i to x_i in advance
    for (int i=1; i<=n; i++) {
      for (int j=1; j<=k; j++) {
        mu_fx[i][i][j] = potentials.potential(i, j);
      }
    }
		
    for (int t=0; t<iterations; t++) {
      // (a) compute messages forward
      compute_mu_forward();
      // (b) compute messages backward
      compute_mu_backward();	
    }
		
    // Compute marginals
    for (int i=1; i<=k; i++) {
      int fkey = n + x_i;
      int fkey0 = fkey==n+1 ? 2*n : fkey-1;
      marginals[i] = mu_fx[fkey0][x_i][i] * mu_fx[fkey][x_i][i] * mu_fx[x_i][x_i][i];
    }
		
    // Normalization
    double sum = 0.0;
    for (int i=1; i<=k; i++) {
      sum += marginals[i];
    }
    for (int i=1; i<=k; i++) {
      marginals[i] = marginals[i]/sum;
    }
		
    return marginals;
  }

}