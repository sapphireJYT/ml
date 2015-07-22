package cs475;

import java.io.Serializable;

public class RegressionLabel extends Label implements Serializable {

	double _regression_label;
	
	public RegressionLabel(double label) {
		this._regression_label = label;
		this._label = String.valueOf(label);
	}

	@Override
	public String toString() {
		return String.valueOf(_regression_label);
	}

}