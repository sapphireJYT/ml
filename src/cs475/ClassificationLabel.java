package cs475;

import java.io.Serializable;

public class ClassificationLabel extends Label implements Serializable {

	public int _classification_label;
	
	public ClassificationLabel(int label) {
		this._classification_label = label;
		this._label = String.valueOf(label);
	}

	@Override
	public String toString() {
		return String.valueOf(this._classification_label);
	}

}