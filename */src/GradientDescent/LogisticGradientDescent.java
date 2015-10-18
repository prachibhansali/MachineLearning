package GradientDescent;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import Jama.Matrix;

public class LogisticGradientDescent {
	double[][] featureMatrix;
	double[][] labels;
	final int ATTR;
	final int INSTANCES;
	final double THRESHOLD;

	public LogisticGradientDescent(int attr,int ins,double t)
	{
		featureMatrix = new double[ins][attr];
		labels = new double[ins][1];
		ATTR = attr;
		INSTANCES = ins;
		THRESHOLD = t;
	}

	public LogisticGradientDescent(double[][] trainset, double[][] trainlabel,
			int a, int i, double t) {
		featureMatrix = trainset;
		labels = trainlabel;
		ATTR = a;
		INSTANCES = i;
		THRESHOLD = t;
	}
	public void readMatrix(String path)
	{
		URL u=null;
		try {
			u = new URL(path);
		} catch (MalformedURLException e1) {

		}
		try {
			read(u.openStream(),"\\s+");
		} catch (IOException e) {

		}
	}

	public void read(InputStream in,String splitStr)
	{
		BufferedReader br = null;
		try{
			br = new BufferedReader(new InputStreamReader(in));
			String line = "";
			int index = 0;
			while((line=br.readLine())!=null)
			{
				String[] attrs = line.trim().split(splitStr);
				if(line.trim().equals("")) break;
				featureMatrix[index][0] = 1;
				for(int i=0;i<attrs.length-1;i++)
					featureMatrix[index][i+1] = Double.parseDouble(attrs[i]);
				labels[index][0]=Double.parseDouble(attrs[attrs.length-1]);
				index++;
			}
		}
		catch(IOException e){
			System.out.println("Invalid file");
		}
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
			for(int j=1;j<INSTANCES;j++)
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
			for(int i=0;i<INSTANCES;i++)
				featureMatrix[i][a] = (double)(featureMatrix[i][a]-mins[a])/(double)maxs[a];
		}
	}

	public double computeSpam(int k,double learnRate) {
		double mse = 0;
		int size = INSTANCES/k;
		double[][] mm = findMinMax();
		performNormalization(mm[0], mm[1]);

		for(int fold = 0;fold<k;fold++)
		{
			Set<Integer> indexes = new HashSet<Integer>();
			randomNumberGenerator(indexes,INSTANCES,size*(k-1));
			double[][] trainset = new double[indexes.size()][ATTR];
			double[][] testset = new double[INSTANCES-indexes.size()+1][ATTR];
			double[][] trainlabel = new double[indexes.size()][1];
			double[][] testlabel = new double[INSTANCES-indexes.size()+1][1];
			createMatrix(featureMatrix,labels,indexes,trainset,testset,trainlabel,testlabel);

			LogisticGradientDescent train = new LogisticGradientDescent(trainset,trainlabel,ATTR,size*(k-1),THRESHOLD);
			LogisticGradientDescent test = new LogisticGradientDescent(testset,testlabel,ATTR,INSTANCES-size*(k-1)+1,THRESHOLD);

			double[][] theta  = train.train(learnRate);
			mse+=test.genAccuracy(theta,fold);
		}
		//System.out.println(mse/(double)k);
		return mse/(double)k;
	}

	double[][] train(double learnRate) {
		double[][] w = new double[ATTR][1];
		Matrix features = new Matrix(featureMatrix);
		Matrix genLabels = applySigmoid(features.times(new Matrix(w)));
		double max=0,prevmax=0;
		int index=0;
		do
		{
			//System.out.println("Iteration #"+index++);
			prevmax=max;
			for(int i=0;i<INSTANCES;i++)
			{
				for(int j=0;j<ATTR;j++)
					w[j][0]=w[j][0]+learnRate*(labels[i][0]-genLabels.get(i,0))*featureMatrix[i][j];
			}
			genLabels = applySigmoid(features.times(new Matrix(w)));
			max = computeLikelihood(genLabels);
		}
		while(Math.abs(prevmax-max)>THRESHOLD);
		return w;
	}

	private double computeLikelihood(Matrix genLabels) {
		double max = 0;
		for(int i=0;i<INSTANCES;i++)
		{
			if(labels[i][0]==1)
				max+=Math.log(genLabels.get(i,0));
			else max+=Math.log(1-genLabels.get(i, 0));
		}
		return max;
	}

	private Matrix applySigmoid(Matrix labels) {
		for(int i=0;i<labels.getRowDimension();i++)
			labels.set(i, 0, 1/(1+Math.exp(-labels.get(i, 0))));
		return labels;
	}

	double genAccuracy(double[][] theta, int fold) {
		int tp=0,tn=0,fp=0,fn=0;
		double x=0,y=0;
		double acc=0;
		double sum=0;
		if(fold==0)
		{
			PrintWriter pw = null;
			try {
				pw = new PrintWriter("roc_linear_logistic");
			} catch (FileNotFoundException e) {}
			
			for(double th=1;th>=0;th-=0.0005)
			{
				double old_x=x,old_y=y;
				tp=0;
				tn=0;
				fp=0;
				fn=0;
				
				Matrix comp = applySigmoid(new Matrix(featureMatrix).times(new Matrix(theta)));
				for(int i=0;i<INSTANCES;i++)
					comp.set(i, 0, comp.get(i, 0) >=th ? 1 : 0);

				for(int i=0;i<INSTANCES;i++)
				{
					double label = comp.get(i, 0);
					if(label==labels[i][0])
					{
						acc++;
						if(label==1) tp++;
						else tn++;
					}
					else {
						if(label==1) fp++;
						else fn++;
					}
				}
				x=(fp/(double)(tn+fp));
				y=(tp/(double)(tp+fn));
				pw.println(x+"\t"+y);
				if(th!=1) sum+=((x-old_x)*(y+old_y));
			}
			System.out.println("Area under the curve "+(0.5*sum));
			acc=acc/(double)50;
			pw.close();
		}
		else {
			Matrix comp = applySigmoid(new Matrix(featureMatrix).times(new Matrix(theta)));
			for(int i=0;i<INSTANCES;i++)
				comp.set(i, 0, comp.get(i, 0) >=0.6 ? 1 : 0);

			for(int i=0;i<INSTANCES;i++)
			{
				double label = comp.get(i, 0);
				if(label==labels[i][0])
				{
					acc++;
					if(label==1) tp++;
					else tn++;
				}
				else {
					if(label==1) fp++;
					else fn++;
				}
			}
		}


		System.out.println("TP TN FP FN "+tp+" "+tn+" "+fp+" "+fn);
		return (acc)/(double)(INSTANCES);
	}
	private void createMatrix(double[][] featureMatrix,double[][] labels,
			Set<Integer> indexes, double[][] train, double[][] test, double[][] trainlabel, double[][] testlabel) {
		int tr=0,ts=0;
		for(int i=0;i<featureMatrix.length;i++)
		{
			if(indexes.contains(i)) {
				trainlabel[tr] = labels[i];
				train[tr++] = featureMatrix[i];				
			}
			else {
				testlabel[ts] = labels[i];
				test[ts++]=featureMatrix[i];
			}
		}
	}

	private void randomNumberGenerator(Set<Integer> indexes, int inst, int size) {
		Random r = new Random();
		while(indexes.size()!=size)
			indexes.add(r.nextInt(inst));
	}

	public static void main(String args[]) throws Exception
	{
		double threshold = 0.00000099;
		double learnRate = 0.000025;
		int spam_attr = 58;
		int spam_inst = 4601;
		int k = 10;
		String spam_path = "/Users/prachibhansali/Documents/Machine Learning/Assignment 1/spambase/";
		LogisticGradientDescent spam_lr = new LogisticGradientDescent(spam_attr,spam_inst,threshold);
		try {
			spam_lr.read(new FileInputStream(spam_path+"spambase.data"),",");
		} catch (FileNotFoundException e) {
			System.out.println("Invalid file");
		}
		System.out.println(spam_lr.computeSpam(k,learnRate));
	}

}
