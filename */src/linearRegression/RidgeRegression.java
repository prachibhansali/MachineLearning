package linearRegression;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.*;

import Jama.Matrix;

public class RidgeRegression {
	//String PATH = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_train.txt";
	double[][] featureMatrix;
	final double[][] labels;
	final int ATTR;
	final int INSTANCES;

	public RidgeRegression(int attr,int ins)
	{
		featureMatrix = new double[ins][attr];
		labels = new double[ins][1];
		ATTR = attr;
		INSTANCES = ins;
	}

	public RidgeRegression(double[][] trainset, double[][] trainlabel, int attr, int ins) {
		featureMatrix = trainset;
		labels = trainlabel;
		ATTR = attr;
		INSTANCES = ins;
	}

	public RidgeRegression(RidgeRegression lr) {
		featureMatrix = new double[lr.featureMatrix.length][lr.featureMatrix[0].length];
		for(int i=0;i<lr.featureMatrix.length;i++)
			for(int j=0;j<lr.featureMatrix[0].length;j++)
				featureMatrix[i][j]=lr.featureMatrix[i][j];
		labels = lr.labels;
		ATTR = lr.ATTR;
		INSTANCES = lr.INSTANCES;
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

	private double computeHousing(int attr, int inst, String path,double r) {
		double[][] mm = findMinMax();
		performNormalization(mm[0], mm[1]);

		RidgeRegression lrtest = new RidgeRegression(attr,506-inst+1);
		path = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_test.txt";
		lrtest.readMatrix(path);
		lrtest.performNormalization(mm[0], mm[1]);

		Matrix theta  = evaluate(r);
		double mse = genMSE(theta,lrtest.featureMatrix,lrtest.labels,lrtest.INSTANCES);
		System.out.println("Housing Test Data MSE = "+mse);
		lrtest = new RidgeRegression(attr,inst);
		path = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_train.txt";
		lrtest.readMatrix(path);
		lrtest.performNormalization(mm[0], mm[1]);

		theta  = evaluate(r);
		mse = genMSE(theta,lrtest.featureMatrix,lrtest.labels,lrtest.INSTANCES);
		System.out.println("Housing Train Data MSE = "+mse);
		return mse;
	}

	public static void main(String args[])
	{
		int attr = 14;
		int inst = 433;
		double r=0.5;
		String path = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_train.txt";
		RidgeRegression lr = new RidgeRegression(attr,inst);
		lr.readMatrix(path);
		System.out.println("Starting spam");
		int spam_attr = 58;
		int spam_inst = 4601;
		int k =10;
		String spam_path = "/Users/prachibhansali/Documents/Machine Learning/Assignment 1/spambase/";
		RidgeRegression spam_lr = new RidgeRegression(spam_attr,spam_inst);
		try {
			spam_lr.read(new FileInputStream(spam_path+"spambase.data"),",");
		} catch (FileNotFoundException e) {
			System.out.println("Invalid file");
		}

		//for(double i=r;i<=0.5;i+=0.01)
		//{
		double h=lr.computeHousing(attr,inst,path,r);
		System.out.println(h);
		double s=spam_lr.computeSpam(spam_attr,spam_inst,k,r);
		//hpw.println(i+"\t"+h);
		//spw.println(i+"\t"+s);
		System.out.println(h+" "+s);
		//}
	}

	private double computeSpam(int attr, int inst,int k,double r) {
		double mse = 0;
		int size = INSTANCES/k;
		double[][] mm = findMinMax();
		performNormalization(mm[0], mm[1]);

		for(int fold = 0;fold<k;fold++)
		{
			Set<Integer> indexes = new HashSet<Integer>();
			randomNumberGenerator(indexes,inst,size*(k-1));

			double[][] trainset = new double[indexes.size()][ATTR];
			double[][] testset = new double[INSTANCES-indexes.size()+1][ATTR];
			double[][] trainlabel = new double[indexes.size()][1];
			double[][] testlabel = new double[INSTANCES-indexes.size()+1][1];
			createMatrix(featureMatrix,labels,indexes,trainset,testset,trainlabel,testlabel);

			RidgeRegression train = new RidgeRegression(trainset,trainlabel,attr,size*(k-1));
			RidgeRegression test = new RidgeRegression(testset,testlabel,attr,INSTANCES-size*(k-1)+1);
			//LinearRegression test = new LinearRegression(trainset,trainlabel,attr,size*(k-1));
			//RidgeRegression test = new RidgeRegression(trainset,trainlabel,attr,size*(k-1));
			Matrix theta  = train.evaluate(r);
			mse+=test.genAccuracy(theta,test.featureMatrix,test.labels,test.INSTANCES);			
		}
		System.out.println("Spam ACC for Training Data "+mse/(double)k);
		return mse/(double)k;
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

	private double genMSE(Matrix theta, double[][] featureMatrix,double[][] labels,int INSTANCES) {
		Matrix comp = new Matrix(featureMatrix).times(theta);
		Matrix error = comp.minus(new Matrix(labels));
		Matrix squaresError = error.transpose().times(error);
		double mse = squaresError.get(0, 0);
		return mse/(double)(INSTANCES);
	}

	private double genAccuracy(Matrix theta, double[][] featureMatrix,double[][] labels,int INSTANCES) {
		int tp=0,tn=0,fp=0,fn=0;
		Matrix comp = new Matrix(featureMatrix).times(theta);
		for(int i=0;i<featureMatrix.length;i++)
			comp.set(i, 0, comp.get(i, 0) >=0.5 ? 1 : 0);

		int acc=0;
		for(int i=0;i<featureMatrix.length;i++)
		{
			if(comp.get(i, 0)==labels[i][0])
				acc++;
			double label = comp.get(i, 0);
			if(label==labels[i][0])
			{
				if(label==1) tp++;
				else tn++;
			}
			else {
				if(label==1) fp++;
				else fn++;
			}
		}
		System.out.println("TP TN FP FN "+tp+" "+tn+" "+fp+" "+fn);
		return acc/(double)(INSTANCES);
	}

	private Matrix evaluate(double r) {
		Matrix features = new Matrix(featureMatrix);
		Matrix label = new Matrix(labels);
		Matrix featureTranspose = features.transpose();
		Matrix ridge = Matrix.identity(featureMatrix[0].length, featureMatrix[0].length).times(r);
		Matrix invFeatures = (featureTranspose.times(features).plus(ridge)).inverse();
		Matrix theta = invFeatures.times(featureTranspose).times(label);
		/*for(int i=0;i<INSTANCES;i++)
			System.out.print(theta.get(i,0)+" ");*/
		return theta;
	}
}
