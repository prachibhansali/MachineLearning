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

public class LinearRegression {
	//String PATH = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_train.txt";
	double[][] featureMatrix;
	double[][] labels;
	final int ATTR;
	final int INSTANCES;

	public LinearRegression(int attr,int ins)
	{
		featureMatrix = new double[ins][attr];
		labels = new double[ins][1];
		ATTR = attr;
		INSTANCES = ins;
	}

	public LinearRegression(double[][] trainset, double[][] trainlabel, int attr, int ins) {
		featureMatrix = trainset;
		labels = trainlabel;
		ATTR = attr;
		INSTANCES = ins;
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
			System.out.println(index+" "+INSTANCES);
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

	private void computeHousing(int attr, int inst, String path) {
		double[][] mm = findMinMax();
		performNormalization(mm[0], mm[1]);

		LinearRegression lrtest = new LinearRegression(attr,506-inst+1);
		path = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_test.txt";
		//LinearRegression lrtest = new LinearRegression(attr,inst);
		//path = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_train.txt";
		lrtest.readMatrix(path);
		lrtest.performNormalization(mm[0], mm[1]);

		Matrix theta  = evaluate();
		double mse = genMSE(theta,lrtest.featureMatrix,lrtest.labels,lrtest.INSTANCES);
		System.out.println(mse);
	}

	public static void main(String args[])
	{
		int attr = 14;
		int inst = 433;
		String path = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_train.txt";
		LinearRegression lr = new LinearRegression(attr,inst);
		lr.readMatrix(path);
		//lr.computeHousing(attr,inst,path);

		attr = 58;
		inst = 4601;
		int k =10;
		path = "/Users/prachibhansali/Documents/Machine Learning/Assignment 1/spambase/";
		lr = new LinearRegression(attr,inst);
		try {
			lr.read(new FileInputStream(path+"spambase.data"),",");
		} catch (FileNotFoundException e) {
			System.out.println("Invalid file");
		}
		lr.computeSpam(attr,inst,k);
	}

	private void computeSpam(int attr, int inst,int k) {
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

			LinearRegression train = new LinearRegression(trainset,trainlabel,attr,size*(k-1));
			LinearRegression test = new LinearRegression(testset,testlabel,attr,INSTANCES-size*(k-1)+1);
			//LinearRegression test = new LinearRegression(trainset,trainlabel,attr,size*(k-1));

			Matrix theta  = train.evaluate();
			mse+=test.genAccuracy(theta.getArray(),fold);			
		}
		System.out.println(mse/(double)k);
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

	public Matrix evaluate() {
		Matrix features = new Matrix(featureMatrix);
		Matrix label = new Matrix(labels);
		Matrix featureTranspose = features.transpose();
		Matrix invFeatures = (featureTranspose.times(features)).inverse();
		Matrix theta = invFeatures.times(featureTranspose).times(label);
		/*for(int i=0;i<INSTANCES;i++)
			System.out.print(theta.get(i,0)+" ");*/
		return theta;
	}

	public double genAccuracy(double[][] theta, int fold) {
		int tp=0,tn=0,fp=0,fn=0;
		double x=0,y=0;
		double acc=0;
		double sum=0;
		if(fold==0)
		{
			PrintWriter pw = null;
			try {
				pw = new PrintWriter("roc_linear_reg_normal");
			} catch (FileNotFoundException e) {}
			
			for(double th=1;th>=0.1;th-=0.02)
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
				if(th!=0) sum+=((x-old_x)*(y+old_y));
			}
			System.out.println("Area under the curve "+(0.5*sum));
			pw.close();
			acc=acc/(double)45;
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

	private Matrix applySigmoid(Matrix labels) {
		for(int i=0;i<labels.getRowDimension();i++)
			labels.set(i, 0, 1/(1+Math.exp(-labels.get(i, 0))));
		return labels;
	}
}
