package regressionTrees;
import java.io.*;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.*;

import decisionTrees.DecisionNode;

public class RegressionTrees {
	String PATH;
	double[][] featureMatrix;
	int ATTR;
	int INSTANCES;
	double THRESHOLD;
	int MINSIZE;
	
	public RegressionTrees(String path,int attr,int ins,double t,int msize)
	{
		featureMatrix = new double[ins][attr];
		PATH = path;
		ATTR = attr;
		INSTANCES = ins;
		THRESHOLD = t;
		MINSIZE = msize;
		createFeatureMatrix();
	}
	
	public void createFeatureMatrix()
	{
		URL url=null;
		try {
			url = new URL(PATH);
		} catch (MalformedURLException e1) {
			System.out.println("Incorrect url");
		}
		BufferedReader br = null;
		try{
			br = new BufferedReader(new InputStreamReader(url.openStream()));
			String line = "";
			int index = 0;
			while((line=br.readLine())!=null)
			{
				String[] attrs = line.trim().split("\\s+");
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
	
	private void evaluate(Node parent) {
		Queue<Node> parentQueue = new LinkedList<Node>();
		Queue<Node> childQueue = new LinkedList<Node>();
		parentQueue.add(parent);
		int height = 1;
		while(!parentQueue.isEmpty())
		{
			Node curr = parentQueue.poll();
			curr.height = height;
			if(!curr.isBelowThreshold(THRESHOLD) && curr.size>MINSIZE)
			{
				Double minError = null;
				int minAttr = -1;
				double minValue = 0;
				Node minleft = null;
				Node minright = null;
				boolean splitPoss = false;
				for(int i=0;i<ATTR-1;i++)
				{
					HashMap<Double,ArrayList<Integer>> mapping = fetchAttributeValueMap(curr.tuples,i);
					ArrayList<Double> uniqueValues = new ArrayList<Double>(mapping.keySet());
					if(uniqueValues.size()<1) continue;
					splitPoss = true;
					Collections.sort(uniqueValues);
					ArrayList<Integer> lefttuple = mapping.get(uniqueValues.get(0));
					Node left = new Node(lefttuple,featureMatrix);
					Node right = new Node(fetchExceptFirst(mapping,uniqueValues),featureMatrix);
					
					for(int j=1;j<uniqueValues.size()-1;j++)
					{
						double error = left.squaredErrorSum+right.squaredErrorSum;
						if(minError==null || error<minError)
						{
							minError = error;
							minAttr = i;
							minleft = new Node(left);
							minright = new Node(right);
							minValue = uniqueValues.get(j-1);
						}
						ArrayList<Integer> t = mapping.get(uniqueValues.get(j));
						left.append(t,featureMatrix);
						right.remove(t.size(),featureMatrix);
					}
					
				}
				if(splitPoss)
				{
					childQueue.add(minleft);
					childQueue.add(minright);
					curr.left = minleft;
					curr.right = minright;
					curr.splitAttr = minAttr;
					curr.splitValue = minValue;
				}
			}
			if(parentQueue.isEmpty())
			{
				parentQueue = new LinkedList<Node>(childQueue);
				childQueue.clear();
				height++;
			}
		}
	}
	
	private HashMap<Double, ArrayList<Integer>> fetchAttributeValueMap(ArrayList<Integer> tuples, int attr) 
	{
		HashMap<Double, ArrayList<Integer>> map = new HashMap<Double, ArrayList<Integer>>();
		for(int i=0;i<tuples.size();i++)
		{
			ArrayList<Integer> tuple = map.containsKey(featureMatrix[tuples.get(i)][attr]) ? map.get(featureMatrix[tuples.get(i)][attr]) : new ArrayList<Integer>();
			tuple.add(tuples.get(i));
			map.put(featureMatrix[tuples.get(i)][attr], tuple);
		}
		return map;
	}

	private ArrayList<Integer> fetchExceptFirst(HashMap<Double,ArrayList<Integer>> mapping,ArrayList<Double> uniqueValues) 
	{
		ArrayList<Integer> right = new ArrayList<Integer>();
		for(int i=1;i<uniqueValues.size();i++)
			right.addAll(mapping.get(uniqueValues.get(i)));
		return right;
	}

	public static void main(String[] args) {
		int attr = 14;
		int inst = 433;
		double t = 0.00001;
		int m = 4;
		RegressionTrees tree = new RegressionTrees("http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_train.txt",attr,inst,t,m);
		ArrayList<Integer> alltups = new ArrayList<Integer>();
		for(int i=0;i<inst;i++)
			alltups.add(i);
		Node parent = new Node(alltups,tree.featureMatrix);
		parent.height=1;
		tree.evaluate(parent);
		
		tree.printGraph(parent);
		RegressionTrees test_tree = new RegressionTrees("http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_test.txt",attr,74,t,m);
		tree.parseTree(parent,test_tree);
	}
	
	private void printGraph(Node parent) {
		Queue<Node> q = new LinkedList<Node>();
		q.add(parent);
		int height = parent.height;
		while(!q.isEmpty())
		{
			Node curr = q.poll();
			if(curr.height!=height)
			{
				//System.out.println();
				//System.out.println();
				height = curr.height;
			}
			//System.out.print(curr.splitAttr+" "+curr.splitValue+"\t");
			//printTuples(curr.tuples);
			//System.out.print("NEXT");
			if(curr.left!=null) q.add(curr.left);
			if(curr.right!=null) q.add(curr.right);
		}
	}
	
	public void printTuples(ArrayList<Integer> arr)
	{
		for(int i=0;i<arr.size();i++)
			System.out.print(arr.get(i)+".");
	}
	
	public void parseTree(Node parent, RegressionTrees test)
	{
		double[][] fm = test.featureMatrix;
		int l = fm[0].length-1;
		double error = 0;
		for(int i=0;i<test.INSTANCES;i++)
		{
			double label = traverseTree(parent,fm[i]);
			error += (fm[i][l]-label)*(fm[i][l]-label);
		}
		System.out.println("Mean Squared Error : "+(double)(error/test.INSTANCES));
	}

	private double traverseTree(Node parent, double[] ds) {
		if(parent==null) return 0;
		int attr = parent.splitAttr;
		double val = parent.splitValue;
		if(ds[attr]<=val) 
		{
			if(parent.left==null)
				return parent.avgLabel();
			else return traverseTree(parent.left, ds);
		}
		else {
			if(parent.right==null)
				return parent.avgLabel();
			else return traverseTree(parent.right, ds);
		}
		
	}
}
