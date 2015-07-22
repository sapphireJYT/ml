package cs475;

import java.io.Serializable;
import java.util.HashMap;

public class FeatureVector implements Serializable {
	
	public HashMap<Integer, Double> _features;
	
	public void add(int index, double value) {
		if (this._features == null) {
			this._features = new HashMap<Integer, Double>();
		}
		this._features.put(index, value);
	}
	
	public double get(int index) {
		if (_features.get(index) != null) {
			return Double.parseDouble(this._features.get(index).toString());
		}
		return 0;
	}

}