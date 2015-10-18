package gradient.discriminant.analysis;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import GradientDescent.GradientDescentClassification;
import Jama.LUDecomposition;
import Jama.Matrix;
import decisionTrees.DecisionNode;
import decisionTrees.DecisionTrees;

public class GDA {
	String PATH = "/Users/prachibhansali/Documents/Machine Learning/Assignment 1/spambase/";
	double[][] featureMatrix;
	double[][] labels;
	final int ATTR;
	final int INSTANCES;
	
	public GDA(int attr,int ins)
	{
		featureMatrix = new double[ins][attr];
		labels = new double[ins][1];
		ATTR = attr;
		INSTANCES = ins;
	}
	
	public GDA(double[][] trainset, double[][] trainlabel,
			int a, int i) {
		featureMatrix = trainset;
		labels = trainlabel;
		ATTR = a;
		INSTANCES = i;
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
				int i=0;
				for(i=0;i<attrs.length-1;i++)
					featureMatrix[index][i] = Double.parseDouble(attrs[i]);
				labels[index][0]=Double.parseDouble(attrs[i]);
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
		for(int i=0;i<ATTR;i++)
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
		for(int a=0;a<ATTR;a++)
		{
			if(maxs[a]==0) continue;
			for(int i=0;i<INSTANCES;i++)
				featureMatrix[i][a] = (double)(featureMatrix[i][a]-mins[a])/(double)maxs[a];
		}
	}
	
	private void randomNumberGenerator(Set<Integer> indexes, int inst, int size) {
		Random r = new Random();
		while(indexes.size()!=size)
			indexes.add(r.nextInt(inst));
	}

	
	private double computeSpam(int attr, int inst,int k) {
		double acc = 0;
		int size = INSTANCES/k;
		double[][] mm = findMinMax();
		performNormalization(mm[0], mm[1]);
		double p1 = probabilityLabelOne(inst,labels);
		double p0 = 1-p1;
		
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

			GDA train = new GDA(trainset,trainlabel,attr,size*(k-1));
			//GDA test = new GDA(testset,testlabel,attr,INSTANCES-size*(k-1)+1);

			GDA test = new GDA(trainset,trainlabel,attr,size*(k-1));
			
			double[] predictedLabels = createGaussianModels(train,test,p0,p1);
			acc+=computeAccuracy(predictedLabels,test.labels);
		}
		//System.out.println(mse/(double)k);
		return acc/(double)k;
	}
	
	private double computeAccuracy(double[] predictedLabels, double[][] labels) {
		
		int count = 0;
		for(int i=0;i<predictedLabels.length;i++)
		{
			if(predictedLabels[i]==labels[i][0]) count++;
		}
		return (count/(double)predictedLabels.length);
	}

	public static double[] createGaussianModels(GDA train,GDA test,double p0,double p1)
	{
		double[][] labelProbability = new double[test.INSTANCES][2];
		
		for(int label=0;label<=1;label++)
		{
			double[][] mean_vector = createMeanVector(train,label);
			double[][] covariance = computeCovariance(train,label,mean_vector);
			
			Matrix cov = new Matrix(covariance);
			double determinant = cov.lu().det();
			cov = cov.inverse();
			
			double gaussian_const = (1/((double)Math.pow((2*Math.PI),(train.ATTR)/2)*Math.sqrt(determinant)));
			
			for(int i=0;i<test.INSTANCES;i++)
			{
				Matrix diff = new Matrix(test.featureMatrix[i],1).minus(new Matrix(mean_vector).transpose());
				Matrix computation = diff.times(cov);
				computation = computation.times(diff.transpose());
				double power = -0.5*computation.get(0, 0);
				labelProbability[i][label] = gaussian_const*Math.exp(power);
			}
		}
		
		double[] predictedLabels = new double[test.INSTANCES];
		for(int i=0;i<test.INSTANCES;i++)
			predictedLabels[i]=(labelProbability[i][0]*p0)>=(labelProbability[i][1]*p1) ? 0 : 1;
		return predictedLabels;
	}
	
	private static double probabilityLabelOne(int instances,double[][] labels) {
		int count = 0;
		for(int i=0;i<instances;i++)
			if(labels[i][0]==1) count++;
		return (count/(double)instances);
	}

	private static double[][] computeCovariance(GDA train, int label,double[][] meanvector) {
		double[][] cov = new double[train.ATTR][train.ATTR];
		for(int row=0;row<train.ATTR;row++)
			for(int col=0;col<train.ATTR;col++)
				cov[row][col]=computeCovValue(train.featureMatrix,train.labels,label,row,col,meanvector);
			
		return cov;
	}

	private static double computeCovValue(double[][] featureMatrix,double[][] labels, int label,int feature1,int feature2,
			double[][] meanvector) {
		double sum = 0;
		int count = 0;
		
		for(int i=0;i<featureMatrix.length;i++)
		{
			if(labels[i][0]!=label) continue;
			sum+=(featureMatrix[i][feature1]-meanvector[feature1][0])*
					(featureMatrix[i][feature2]-meanvector[feature2][0]);
			count++;
		}
		return sum/(double) count;
	}

	private static double[][] createMeanVector(GDA train, int label) {
		double[][] mean = new double[train.ATTR][1];
		
		for(int i=0;i<train.ATTR;i++)
		{
			double sum = 0;
			int count =0;
			for(int ins = 0;ins<train.INSTANCES;ins++)
			{
				if(train.labels[ins][0]!=label) continue;
				sum+=train.featureMatrix[ins][i];
				count++;
			}
			mean[i][0]=sum/(double)count;
		}
		return mean;
	}

	private static void createMatrix(double[][] featureMatrix,double[][] labels,
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
	
	public static void main(String args[])
	{
		int attr = 57;
		int inst = 4601;
		int k = 10;
		
		GDA spam = new GDA(attr,inst);
		spam.createFeatureMatrix();
		System.out.println(spam.computeSpam(attr,inst,k));
	}

}
