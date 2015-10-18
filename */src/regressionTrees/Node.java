package regressionTrees;

import java.util.ArrayList;

public class Node {
	ArrayList<Integer> tuples;
	double squaredErrorSum;
	double sumLabel;
	int size;
	Node left;
	Node right;
	int height;
	int splitAttr;
	double splitValue;

	public Node()
	{
		tuples = new ArrayList<Integer>();
	}
	
	public Node(Node n)
	{
		this.tuples = new ArrayList<Integer>(n.tuples);
		this.squaredErrorSum=n.squaredErrorSum;
		this.sumLabel=n.sumLabel;
		this.size=n.size;
		this.splitAttr = n.splitAttr;
		this.splitValue = n.splitValue;
	}
	
	public Node(ArrayList<Integer> tuples,double[][] featureMatrix)
	{
		this.tuples = new ArrayList<Integer>(tuples);
		size = tuples.size();
		int label = featureMatrix[0].length-1;
		
		genSumLabel(tuples,label,featureMatrix);
		getSquaredError(label,featureMatrix);
	}

	private void getSquaredError(int label,double[][] featureMatrix) 
	{
		squaredErrorSum = 0;
		for(int i=0;i<tuples.size();i++)
		{
			double e = (featureMatrix[tuples.get(i)][label]-avgLabel());
			squaredErrorSum += e*e;
		}
	}

	private void genSumLabel(ArrayList<Integer> tuples,int label, double[][] featureMatrix) 
	{
		for(int i=0;i<tuples.size();i++)
			sumLabel+=featureMatrix[tuples.get(i)][label];
	}

	private void reduceSumLabel(int toRemove,int label, double[][] featureMatrix) 
	{
		for(int i=0;i<toRemove;i++)
			sumLabel-=featureMatrix[tuples.get(i)][label];
	}
	
	public ArrayList<Integer> getTuples() {
		return tuples;
	}

	public void setTuples(ArrayList<Integer> tuples) {
		this.tuples = tuples;
	}

	public boolean isBelowThreshold(double t)
	{
		return errorThreshold()<t;			
	}
	
	public double avgLabel()
	{
		return sumLabel/size;
	}

	public double errorThreshold()
	{
		return (squaredErrorSum/((double)tuples.size()));
	}

	public void append(ArrayList<Integer> t,double[][] featureMatrix) 
	{
		int label = featureMatrix[0].length-1;
		genSumLabel(t,label,featureMatrix);
		size+=t.size();
		tuples.addAll(t);
		getSquaredError(label,featureMatrix);
	}

	public void remove(int toRemove,double[][] featureMatrix) {
		int label = featureMatrix[0].length-1;
		reduceSumLabel(toRemove,label,featureMatrix);
		size-=toRemove;
		while(toRemove!=0)
		{
			tuples.remove(0);
			toRemove--;
		}
		getSquaredError(label,featureMatrix);
	}
}
