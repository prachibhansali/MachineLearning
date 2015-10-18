package perceptron;
import java.io.*;
import java.util.Arrays;

import Jama.Matrix;

public class Perceptron {

	double[][] featureMatrix;
	double[][] labels;
	final int ATTR;
	final int INST;

	public Perceptron(int attr,int inst)
	{
		featureMatrix = new double[inst][attr];
		ATTR = attr;
		INST = inst;
		labels = new double[inst][1];
	}

	public void readInput(String path)
	{
		BufferedReader br = null;
		try{
			br = new BufferedReader(new FileReader(path));
		}
		catch(IOException e)
		{
			System.out.println("File not found");
		}

		String line="";
		try {
			int index=0;
			while((line=br.readLine())!=null)
			{
				String[] attrs = line.split("\\s+");
				featureMatrix[index][0]=1;
				for(int i=0;i<attrs.length-1;i++)
					featureMatrix[index][i+1]=Double.parseDouble(attrs[i]);
				labels[index][0]=Double.parseDouble(attrs[ATTR-1]);
				index++;
			}
		} catch (IOException e1) {
			System.out.println("Incorrect input");
		}

		try {
			br.close();
		} catch (IOException e) {
			System.out.println("File not closed");
		}

		double[][] mm = findMinMax();
		performNormalization(mm[0],mm[1]);
	}

	public double[][] findMinMax()
	{
		double[] mins = new double[ATTR];
		double[] maxs = new double[ATTR];
		mins[0]=0;
		for(int i=1;i<ATTR;i++)
		{
			double min=featureMatrix[0][i];
			double max=featureMatrix[0][i];
			for(int j=1;j<INST;j++)
			{
				min = Double.min(min,featureMatrix[j][i]);
				max = Double.max(max,featureMatrix[j][i]);
			}
			max = max-min;
			mins[i]=min;
			maxs[i]=max;
		}
		return new double[][]{mins,maxs};
	}

	public void performNormalization(double[]mins,double[]maxs)
	{
		for(int a=1;a<ATTR;a++)
		{
			if(maxs[a]==0) continue;
			for(int i=0;i<INST;i++)
				featureMatrix[i][a] = (double)(featureMatrix[i][a]-mins[a])/(double)maxs[a];
		}
	}

	private double[][] train(double learnRate) 
	{
		for(int i=0;i<INST;i++)
			if(labels[i][0]<0)
				for(int f=0;f<ATTR;f++)
					featureMatrix[i][f]=-featureMatrix[i][f];
		
		double[][] w = new double[ATTR][1];
		w[0][0]=1;
		Matrix wm = new Matrix(w);
		int index=0,count=0;
		boolean seen = false;
		do
		{
			seen=false;
			Matrix hx = (new Matrix(featureMatrix)).times(wm);
			count=0;
			for(int i=0;i<hx.getRowDimension();i++)
			{
				if(hx.get(i, 0)<0)
				{
					count++;
					if(!seen) seen=true;
					double[][] nmatr = new double[1][ATTR];
					nmatr[0]=featureMatrix[i];
					wm = wm.plus(new Matrix(nmatr).transpose().times(learnRate));
				}
			}
			System.out.println("Iteration # "+index++ +" , total_mistake "+count);
			//System.out.println(Arrays.toString(wm.getColumnPackedCopy())+" "+count);
		}
		while(seen);
		System.out.println("Classifier Weights: "+Arrays.toString(wm.getColumnPackedCopy()));
		double w0=wm.get(0, 0);
		double[] vals = new double[ATTR-1];
		for(int i=1;i<wm.getRowDimension();i++)
		{
			vals[i-1]=wm.get(i,0)/-w0;
		}
		System.out.println("Normalized with threshold: "+Arrays.toString(vals));
		return w;
	}

	public static void main(String args[])
	{
		int attr=5;
		int inst = 1000;
		double learnRate=20;
		Perceptron p = new Perceptron(attr,inst);
		p.readInput("/Users/prachibhansali/Documents/Machine Learning/Assignment 2/perceptronData.txt");
		p.train(learnRate);		
	}
}
