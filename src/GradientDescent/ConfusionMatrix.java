package GradientDescent;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import Jama.Matrix;
import linearRegression.LinearRegression;

public class ConfusionMatrix {
	double[][] featureMatrix;
	double[][] labels;
	final int ATTR;
	final int INSTANCES;
	final double THRESHOLD;

	public ConfusionMatrix(int attr,int ins,double t)
	{
		featureMatrix = new double[ins][attr];
		labels = new double[ins][1];
		ATTR = attr;
		INSTANCES = ins;
		THRESHOLD = t;
	}
	
	public ConfusionMatrix(double[][] trainset, double[][] trainlabel,
			int a, int i, double t) {
		featureMatrix = trainset;
		labels = trainlabel;
		ATTR = a;
		INSTANCES = i;
		THRESHOLD = t;
	}
	
	public static void main(String args[])
	{
		double threshold = 0.00099;
		double learnRate = 0.0025;
		int spam_attr = 58;
		int spam_inst = 4601;
		int k = 10;
		String spam_path = "/Users/prachibhansali/Documents/Machine Learning/Assignment 1/spambase/";
		ConfusionMatrix spam_lr = new ConfusionMatrix(spam_attr,spam_inst,threshold);
		try {
			spam_lr.read(new FileInputStream(spam_path+"spambase.data"),",");
		} catch (FileNotFoundException e) {
			System.out.println("Invalid file");
		}
		
		ConfusionMatrix[] tt = spam_lr.computeSpam(k,learnRate); 
		ConfusionMatrix train = tt[0];
		ConfusionMatrix test = tt[1];
		
		LinearRegression ltrain = new LinearRegression(train.featureMatrix.clone(),train.labels.clone(),train.ATTR,train.INSTANCES);
		LinearRegression ltest = new LinearRegression(test.featureMatrix.clone(),test.labels.clone(),test.ATTR,test.INSTANCES);
		Matrix theta  = ltrain.evaluate();
		double mse = ltest.genAccuracy(theta.getArray(),0);
		
		GradientDescentClassification gdtrain = new GradientDescentClassification(train.featureMatrix.clone(),train.labels.clone(),train.ATTR,train.INSTANCES,threshold);
		GradientDescentClassification gdtest = new GradientDescentClassification(test.featureMatrix.clone(),test.labels.clone(),test.ATTR,test.INSTANCES,threshold);
		double[][] theta_arr  = gdtrain.trainClassification(gdtrain.ATTR,gdtrain.INSTANCES,learnRate,threshold);
		mse = gdtest.genAccuracy(theta_arr,0);
		
		LogisticGradientDescent lgtrain = new LogisticGradientDescent(train.featureMatrix.clone(),train.labels.clone(),train.ATTR,train.INSTANCES,threshold);
		LogisticGradientDescent lgtest = new LogisticGradientDescent(test.featureMatrix.clone(),test.labels.clone(),test.ATTR,test.INSTANCES,threshold);
		theta_arr  = lgtrain.train(learnRate);
		mse = lgtest.genAccuracy(theta_arr,0);
		
		
	}
	
	public ConfusionMatrix[] computeSpam(int k,double learnRate) {
		double mse = 0;
		int size = INSTANCES/k;
		double[][] mm = findMinMax();
		performNormalization(mm[0], mm[1]);

		Set<Integer> indexes = new HashSet<Integer>();
			randomNumberGenerator(indexes,INSTANCES,size*(k-1));
			double[][] trainset = new double[indexes.size()][ATTR];
			double[][] testset = new double[INSTANCES-indexes.size()+1][ATTR];
			double[][] trainlabel = new double[indexes.size()][1];
			double[][] testlabel = new double[INSTANCES-indexes.size()+1][1];
			createMatrix(featureMatrix,labels,indexes,trainset,testset,trainlabel,testlabel);

			ConfusionMatrix train = new ConfusionMatrix(trainset,trainlabel,ATTR,size*(k-1),THRESHOLD);
			ConfusionMatrix test = new ConfusionMatrix(testset,testlabel,ATTR,INSTANCES-size*(k-1)+1,THRESHOLD);
			return new ConfusionMatrix[]{train,test};
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
}
