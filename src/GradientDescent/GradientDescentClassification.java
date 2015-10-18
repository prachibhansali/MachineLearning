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


public class GradientDescentClassification {
	double[][] featureMatrix;
	double[][] labels;
	final int ATTR;
	final int INSTANCES;
	final double THRESHOLD;

	public GradientDescentClassification(int attr,int ins,double t)
	{
		featureMatrix = new double[ins][attr];
		labels = new double[ins][1];
		ATTR = attr;
		INSTANCES = ins;
		THRESHOLD = t;
	}

	public GradientDescentClassification(double[][] trainset, double[][] trainlabel,
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

	public double[][] trainClassification(int attr, int inst, double learnRate,double threshold) {
		double w[][] = new double[featureMatrix[0].length][1];
		double prevError=0,error=0;
		int in=0;
		do
		{
			prevError = error;
			double[] genLabels = computeDerivSumClass(w);
			//System.out.println(Arrays.toString(genLabels));
			for(int ins=0;ins<INSTANCES;ins++)
				for(int i=0;i<w.length;i++)
					w[i][0]=w[i][0]-(learnRate*(genLabels[ins]-labels[ins][0])*genLabels[ins]*(1-genLabels[ins])*featureMatrix[ins][i]);
			error = evaluateClassification(w);
		}
		while(!converges(prevError,error));
		return w;
	}

	private boolean converges(double prevError, double error) {
		return Math.abs(error-prevError)<THRESHOLD;
	}
	
	public double evaluateClassification(double[][] w) {
		Matrix features = new Matrix(featureMatrix);
		Matrix label = new Matrix(labels);
		Matrix theta = new Matrix(w);

		Matrix error = convertUsingSigmoid(features.times(theta)).minus(label);
		double squaredError = error.transpose().times(error).get(0, 0);
		double e = (0.5*squaredError)/INSTANCES;
		//System.out.println(Arrays.toString(theta.getRowPackedCopy()));
		return e;
	}

	private Matrix convertUsingSigmoid(Matrix h) {
		for(int i=0;i<featureMatrix.length;i++)
		{
			h.set(i, 0, 1/(1+Math.exp(-h.get(i,0))));
		}
		return h;
	}

	private double[] computeDerivSumClass(double[][] w) {
		Matrix h = convertUsingSigmoid(new Matrix(featureMatrix).times(new Matrix(w)));
		//System.out.println(Arrays.toString(h.getColumnPackedCopy()));

		return h.getColumnPackedCopy();
	}

	public static void main(String args[]) throws Exception
	{
		double threshold = 0.0000099;
		double learnRate = 0.000212;
		int spam_attr = 58;
		int spam_inst = 4601;
		int k = 10;
		String spam_path = "/Users/prachibhansali/Documents/Machine Learning/Assignment 1/spambase/";
		GradientDescentClassification spam_lr = new GradientDescentClassification(spam_attr,spam_inst,threshold);
		try {
			spam_lr.read(new FileInputStream(spam_path+"spambase.data"),",");
		} catch (FileNotFoundException e) {
			System.out.println("Invalid file");
		}
		System.out.println(spam_lr.computeSpam(spam_attr,spam_inst,k,learnRate));
	}

	private double computeSpam(int attr, int inst,int k,double learnRate) {
		double mse = 0;
		int size = INSTANCES/k;
		double[][] mm = findMinMax();
		performNormalization(mm[0], mm[1]);

		for(int fold = 0;fold<k;fold++)
		{
			Set<Integer> indexes = new HashSet<Integer>();
			randomNumberGenerator(indexes,inst,size*(k-1));
			//System.out.println("fold #");
			double[][] trainset = new double[indexes.size()][ATTR];
			double[][] testset = new double[INSTANCES-indexes.size()+1][ATTR];
			double[][] trainlabel = new double[indexes.size()][1];
			double[][] testlabel = new double[INSTANCES-indexes.size()+1][1];
			createMatrix(featureMatrix,labels,indexes,trainset,testset,trainlabel,testlabel);

			GradientDescentClassification train = new GradientDescentClassification(trainset,trainlabel,attr,size*(k-1),THRESHOLD);
			GradientDescentClassification test = new GradientDescentClassification(testset,testlabel,attr,INSTANCES-size*(k-1)+1,THRESHOLD);

			//LinearRegression test = new LinearRegression(trainset,trainlabel,attr,size*(k-1));

			double[][] theta  = train.trainClassification(attr, inst, learnRate, THRESHOLD);
			mse+=test.genAccuracy(theta,fold);
		}
		//System.out.println(mse/(double)k);
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


	public double genAccuracy(double[][] theta, int fold) {
		int tp=0,tn=0,fp=0,fn=0;
		double x=0,y=0;
		
		double sum=0;
		if(fold==0)
		{
			PrintWriter pw = null;
			try {
				pw = new PrintWriter("roc_linear_gd");
			} catch (FileNotFoundException e) {}
			
			for(double th=1;th>=0;th-=0.002)
			{
				double old_x=x,old_y=y;
				tp=0;
				tn=0;
				fp=0;
				fn=0;
				Matrix comp = convertUsingSigmoid(new Matrix(featureMatrix).times(new Matrix(theta)));
				for(int i=0;i<INSTANCES;i++)
					comp.set(i, 0, comp.get(i, 0) >=th ? 1 : 0);

				for(int i=0;i<INSTANCES;i++)
				{
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
				x=(fp/(double)(tn+fp));
				y=(tp/(double)(tp+fn));
				pw.println(x+"\t"+y);
				if(th!=0) sum+=((x-old_x)*(y+old_y));
			}
			System.out.println("Area under the curve "+(0.5*sum));
			pw.close();
		}
		else {
			Matrix comp = convertUsingSigmoid(new Matrix(featureMatrix).times(new Matrix(theta)));
			for(int i=0;i<INSTANCES;i++)
				comp.set(i, 0, comp.get(i, 0) >=0.6 ? 1 : 0);

			for(int i=0;i<INSTANCES;i++)
			{
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
		}


		System.out.println("TP TN FP FN "+tp+" "+tn+" "+fp+" "+fn);
		return (tn+tp)/(double)(INSTANCES);
	}


}
