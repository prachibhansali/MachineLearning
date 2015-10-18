package decisionTrees;

import java.io.*;
import java.util.*;

import decisionTrees.DecisionNode;

public class DecisionTrees {
	String PATH = "/Users/prachibhansali/Documents/Machine Learning/Assignment 1/spambase/";
	double[][] featureMatrix;
	final int ATTR;
	final int INSTANCES;
	final double THRESHOLD;
	final int MINSIZE;

	public DecisionTrees(int attr,int ins,double t,int msize)
	{
		featureMatrix = new double[ins][attr];
		ATTR = attr;
		INSTANCES = ins;
		THRESHOLD = t;
		MINSIZE = msize;
	}

	public void createFeatureMatrix()
	{
		BufferedReader br = null;
		try{
			br = new BufferedReader(new FileReader(PATH+"spambase.data"));
			String line = "";
			int index = 0;
			while((line=br.readLine())!=null)
			{
				String[] attrs = line.trim().split(",");
				if(line.equals("")) break;
				for(int i=0;i<attrs.length;i++)
					featureMatrix[index][i] = Double.parseDouble(attrs[i]);
				index++;
			}
			System.out.println(index+" "+INSTANCES);
		}
		catch(IOException e){
			System.out.println("Invalid file");
		}
	}

	private void evaluate(DecisionNode parent) {
		Queue<DecisionNode> parentQueue = new LinkedList<DecisionNode>();
		Queue<DecisionNode> childQueue = new LinkedList<DecisionNode>();
		parentQueue.add(parent);
		int height = 1;
		while(!parentQueue.isEmpty())
		{
			DecisionNode curr = parentQueue.poll();
			curr.height = height;
			if(!curr.isBelowThreshold(THRESHOLD) && !((curr.size)<MINSIZE))
			{
				DecisionNode[] nodes = chooseBestAttribute(curr);
				if(nodes!=null)
				{
					curr.left = nodes[0];
					curr.right = nodes[1];
					childQueue.add(nodes[0]);
					childQueue.add(nodes[1]);
				}
			}
			if(parentQueue.isEmpty())
			{
				parentQueue = new LinkedList<DecisionNode>(childQueue);
				childQueue.clear();
				height++;
			}
		}
	}

	private DecisionNode[] chooseBestAttribute(DecisionNode curr) {
		DecisionNode minleft = null;
		DecisionNode minright = null;
		Integer minAttr = null;
		Double maxGain = null;
		Double attrValue = null;
		
		for(int i=0;i<ATTR-1;i++)
		{
			HashMap<Double,ArrayList<Integer>> h = createAttrMapping(curr.tuples,i);
			ArrayList<Double> distinctValues = new ArrayList<Double>(h.keySet());
			if(distinctValues.size() <= 1) continue; 
			Collections.sort(distinctValues);
			ArrayList<Integer> left = new ArrayList<Integer>(h.get(distinctValues.get(0)));
			DecisionNode leftDecision = new DecisionNode(left,featureMatrix);
			DecisionNode rightDecision = new DecisionNode(fetchExceptFirst(h,distinctValues),featureMatrix);
			
			for(int j=1;j<distinctValues.size()-1;j++)
			{
				double infoGain = curr.entropy() - ((leftDecision.size*leftDecision.entropy()+rightDecision.size*rightDecision.entropy())/curr.size);
				
				if(infoGain>0 && (maxGain==null || infoGain > maxGain))
				{
					maxGain = infoGain;
					minleft = new DecisionNode(leftDecision);
					minright = new DecisionNode(rightDecision);
					minAttr = i;
					attrValue = distinctValues.get(j-1);
				}
				leftDecision.append(h.get(distinctValues.get(j)),featureMatrix);
				rightDecision.remove(h.get(distinctValues.get(j)).size(),featureMatrix);
			}
		}
		if(minAttr==null) return null;
		curr.splitAttr = minAttr;
		curr.splitValue = attrValue;
		return new DecisionNode[]{minleft,minright};
	}

	private HashMap<Double, ArrayList<Integer>> createAttrMapping(
			ArrayList<Integer> tuples,int attr) {
		HashMap<Double, ArrayList<Integer>> h = new HashMap<Double, ArrayList<Integer>>();
		for(int i=0;i<tuples.size();i++)
		{
			double val = featureMatrix[tuples.get(i)][attr];
			ArrayList<Integer> tups = h.containsKey(val) ? h.get(val) : new ArrayList<Integer>();
			tups.add(tuples.get(i));
			h.put(val,tups);
		}
		return h;
	}

	private ArrayList<Integer> fetchExceptFirst(HashMap<Double,ArrayList<Integer>> mapping,ArrayList<Double> uniqueValues) 
	{
		ArrayList<Integer> right = new ArrayList<Integer>();
		for(int i=1;i<uniqueValues.size();i++)
			right.addAll(mapping.get(uniqueValues.get(i)));
		return right;
	}

	public static void main(String[] args) {
		int attr = 58;
		double t = 0.0005;
		int m = 4;
		int inst = 4601;
		int k = 10;
		int size = inst/k;
		double mse = 0;
		DecisionTrees tree = new DecisionTrees(attr,inst,t,m);
		tree.createFeatureMatrix();

		for(int fold=0;fold<k;fold++)
		{
			ArrayList<Integer> alltups = new ArrayList<Integer>();
			for(int i=0;i<inst;i++)
			{
				if(i>=(fold*size) && i<=(fold*size+size)) continue;
				alltups.add(i);
			}
			DecisionNode parent = new DecisionNode(alltups,tree.featureMatrix);
			parent.height=1;
			tree.evaluate(parent);

			//tree.printDecisionGraph(parent);
			mse += tree.parseTree(parent,fold,size);
		}
		System.out.println("Mean Squared Error = "+(double)(mse/(double)k));
	}

	public void printTuples(ArrayList<Integer> arr)
	{
		for(int i=0;i<arr.size();i++)
			System.out.print(arr.get(i)+".");
	}

	public double parseTree(DecisionNode parent, int fold,int size)
	{
		int fp=0,tp=0,fn=0,tn=0;
		int l = featureMatrix[0].length-1;
		//for(int i=fold*size;i<=(fold*size)+size;i++)
		//{
		int index = 0;
		for(int i=0;i<featureMatrix.length;i++)
		{
			if(i>=(fold*size) && i<=(fold*size+size)); 
			{
			double label = traverseTree(parent,featureMatrix[i]);
			if(label==featureMatrix[i][l])
			{
				if(label==1) tp++;
				else tn++;
			}
			else {
				if(label==1) fp++;
				else fn++;
			}
			index++;
			}
		}
		System.out.println("Error in train data "+(fold+1)+": "+((fp+fn)/(double)(index+1)));
		System.out.println("TP TN FP FN "+tp+" "+tn+" "+fp+" "+fn); // TP TN FP FN 377 26 2 56
		return ((fp+fn)/(double)(index+1));
	}

	private int traverseTree(DecisionNode parent, double[] ds) {
		if(parent==null) return 0;
		int attr = parent.splitAttr;
		double val = parent.splitValue;
		if(ds[attr]<=val) 
		{
			if(parent.left==null)
				return parent.spam >= (parent.size/2) ? 1 : 0;
				else return traverseTree(parent.left, ds);
		}
		else {
			if(parent.right==null)
				return parent.spam >= (parent.size/2) ? 1 : 0;
				else return traverseTree(parent.right, ds);
		}

	}

}
