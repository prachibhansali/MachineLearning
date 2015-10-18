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
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import Jama.Matrix;
import linearRegression.LinearRegression;


public class GradientDescent {
	double[][] featureMatrix;
	double[][] labels;
	final int ATTR;
	final int INSTANCES;
	final double THRESHOLD;

	public GradientDescent(int attr,int ins,double t)
	{
		featureMatrix = new double[ins][attr];
		labels = new double[ins][1];
		ATTR = attr;
		INSTANCES = ins;
		THRESHOLD = t;
	}

	public GradientDescent(double[][] trainset, double[][] trainlabel,
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

	private double[][] train(int attr, int inst, double learnRate,double threshold) {
		double w[][] = new double[featureMatrix[0].length][1];
		double prevError=0,error=0;
		do
		{
			prevError = error;
			double[] genLabels = computeDerivSum(w);
			//System.out.println(Arrays.toString(genLabels));
			for(int i=0;i<w.length;i++)
			{
				w[i][0]=w[i][0]-learnRate*sumTrainingData(genLabels,i);
			}
			error = evaluate(w);
		}
		while(!converges(prevError,error));
		return w;
	}

	private double predict(double[][] w) {
		Matrix comp = new Matrix(featureMatrix).times(new Matrix(w));
		Matrix error = comp.minus(new Matrix(labels));
		Matrix squaresError = error.transpose().times(error);
		double mse = squaresError.get(0, 0);
		return mse/(double)(INSTANCES);
	}

	private boolean converges(double prevError, double error) {
		return Math.abs(error-prevError)<THRESHOLD;
	}

	private double evaluate(double[][] w) {
		Matrix features = new Matrix(featureMatrix);
		Matrix label = new Matrix(labels);
		Matrix theta = new Matrix(w);

		Matrix error = features.times(theta).minus(label);
		double e = (error.transpose().times(error)).get(0, 0)/INSTANCES;
		//System.out.println(Arrays.toString(theta.getRowPackedCopy()));
		return e;
	}

	private double sumTrainingData(double[] genLabels, int wth) {
		double sum=0;
		for(int i=0;i<featureMatrix.length;i++)
			sum+=(-labels[i][0]+genLabels[i])*featureMatrix[i][wth];
		return sum;
	}

	private double[] computeDerivSum(double[][] w) {
		Matrix h = new Matrix(featureMatrix).times(new Matrix(w));
		//System.out.println(Arrays.toString(h.getColumnPackedCopy()));

		return h.getColumnPackedCopy();
	}

	public static void main(String args[]) throws Exception
	{
		int attr = 14;
		int inst = 433;
		double threshold = 0.0001;
		double learnRate = 0.000299;
		String path = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_train.txt";
		GradientDescent lr = new GradientDescent(attr,inst,threshold);
		lr.readMatrix(path);

		double[][] mm = lr.findMinMax();
		lr.performNormalization(mm[0],mm[1]);
		double[][]w = lr.train(attr, inst, learnRate,threshold);

		GradientDescent test = lr.createTestingData(mm[0],mm[1]);
		System.out.println(test.predict(w));
	}



	private GradientDescent createTestingData(double[] min,double[] max) {
		GradientDescent lrtest = new GradientDescent(ATTR,506-INSTANCES+1,THRESHOLD);
		//GradientDescent lrtest = new GradientDescent(attr,inst,threshold);
		String path = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_test.txt";
		//LinearRegression lrtest = new LinearRegression(attr,inst);
		//path = "http://www.ccs.neu.edu/home/vip/teach/MLcourse/data/housing_train.txt";
		lrtest.readMatrix(path);
		lrtest.performNormalization(min,max);
		return lrtest;
	}



}
