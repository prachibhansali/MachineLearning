package decisionTrees;

import java.util.ArrayList;

public class DecisionNode {
	ArrayList<Integer> tuples;
	int size;
	DecisionNode left;
	DecisionNode right;
	int height;
	int splitAttr;
	double splitValue;
	int spam; // -plogp
	
	public DecisionNode(ArrayList<Integer> t,double[][] featureMatrix)
	{
		tuples = new ArrayList<Integer>(t);
		size = t.size();
		calculateSpam(tuples,featureMatrix);
	}
	
	public DecisionNode(DecisionNode n) {
		this.tuples = new ArrayList<Integer>(n.tuples);
		this.spam=n.spam;
		this.size=n.size;
		this.splitAttr = n.splitAttr;
		this.splitValue = n.splitValue;
	}

	public boolean isBelowThreshold(double t)
	{
		return entropy()<t;
	}
	
	public void calculateSpam(ArrayList<Integer> tuples,double[][] featureMatrix)
	{
		int l = featureMatrix[0].length-1;
		for(int i=0;i<tuples.size();i++)
			if(featureMatrix[tuples.get(i)][l]==1) 
				spam++;
 	}
	
	public void remove(int s, double[][] featureMatrix) {
		int l = featureMatrix[0].length-1;
		for(int i=0;i<s;i++)
		{
			if(featureMatrix[tuples.get(0)][l]==1) spam--;
			tuples.remove(0);
		}
		size-=s;
	}
	
	public double entropy()
	{
		if(spam==0) return 0;
		double pspam = spam/(double)size;
		return -pspam*Math.log(pspam) - (1-pspam)*Math.log(1-pspam);
	}

	public void append(ArrayList<Integer> t, double[][] featureMatrix) {
		calculateSpam(t,featureMatrix);
		tuples.addAll(t);
		size+=t.size();
	}
	
}
