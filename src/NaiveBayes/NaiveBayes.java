package NaiveBayes;

import gradient.discriminant.analysis.GDA;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class NaiveBayes {
	String PATH = "/Users/prachibhansali/Documents/Machine Learning/Assignment 1/spambase/";
	double[][] featureMatrix;
	double[][] labels;
	final int ATTR;
	final int INSTANCES;

	public NaiveBayes(int attr,int ins)
	{
		featureMatrix = new double[ins][attr];
		labels = new double[ins][1];
		ATTR = attr;
		INSTANCES = ins;
	}

	public NaiveBayes(double[][] trainset, double[][] trainlabel,
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

	private void chooseTenthElements(Set<Integer> indexes, int inst, int fold) {
		for(int i=fold;i<inst;i+=10)
			indexes.add(i);
	}


	private double[] computeSpam(int attr, int inst,int k,double alpha,double galpha, int bins) {

		double[][] mm = findMinMax();
		performNormalization(mm[0], mm[1]);
		double spamPrior=probabilityLabelOne(INSTANCES,labels);
		double nonspamPrior =1-spamPrior;
		double[] acc = new double[3];

		for(int fold = 0;fold<k;fold++)
		{
			Set<Integer> indexes = new HashSet<Integer>();
			chooseTenthElements(indexes,inst,fold);
			//System.out.println("fold #");
			double[][] trainset = new double[INSTANCES-indexes.size()+1][ATTR];
			double[][] testset = new double[indexes.size()][ATTR];
			double[][] trainlabel = new double[INSTANCES-indexes.size()+1][1];
			double[][] testlabel = new double[indexes.size()][1];
			createMatrix(featureMatrix,labels,indexes,trainset,testset,trainlabel,testlabel);

			NaiveBayes train = new NaiveBayes(trainset,trainlabel,attr,(INSTANCES-indexes.size()+1));
			NaiveBayes test = new NaiveBayes(testset,testlabel,attr,indexes.size());
			int index = 0;
			//GDA test = new GDA(trainset,trainlabel,attr,size*(k-1));			
			double[] predictedLabels = performNaiveBayesUsingBernoulli(train,test,alpha,spamPrior,nonspamPrior);
			acc[index]+=computeAccuracy(predictedLabels,test.labels);
			index++;

			predictedLabels = performNaiveBayesUsingGaussian(train,test,galpha,spamPrior,nonspamPrior);
			acc[index]+=computeAccuracy(predictedLabels,test.labels);
			index++;

			predictedLabels = performHistogramDistribution(train,test,alpha,spamPrior,nonspamPrior,bins);
			acc[index]+=computeAccuracy(predictedLabels,test.labels);
			index++;
		}
		//System.out.println(mse/(double)k);
		for(int i=0;i<acc.length;i++)
			acc[i]/=k;
		return acc;
	}


	private double[] performHistogramDistribution(
			NaiveBayes train, NaiveBayes test,
			double alpha, double spamPrior, double nonspamPrior,int bins) {
		double[][] ranges = new double[train.ATTR][bins+1];
		double[] means = computeMeanVectors(train);

		for(int attr=0;attr<train.ATTR;attr++)
		{
			double[] minmax=findMinMaxAttribute(train.featureMatrix,train.labels,attr,alpha);
			ranges[attr][0]=minmax[0];
			ranges[attr][bins]=minmax[1]+alpha;
			if(bins==4)
			{
				ranges[attr][1]=Double.min(minmax[2], minmax[3]);
				ranges[attr][3]=Double.max(minmax[2], minmax[3]);
				ranges[attr][2]=means[attr];
			}

			if(bins==9)
			{
				ranges[attr][2]=Double.min(minmax[2], minmax[3]);
				ranges[attr][6]=Double.max(minmax[2], minmax[3]);
				ranges[attr][4]=means[attr];
				ranges[attr][1]=(ranges[attr][0]+ranges[attr][2])/2;
				ranges[attr][3]=(ranges[attr][2]+ranges[attr][4])/2;
				ranges[attr][5]=(ranges[attr][4]+ranges[attr][6])/2;
				ranges[attr][7]=(2*ranges[attr][6]+ranges[attr][9])/3;
				ranges[attr][8]=(2*ranges[attr][6]+2*ranges[attr][9])/3;
				//createPartitions(means[attr],ranges,bins,train.featureMatrix,attr);
			}
			System.out.println(Arrays.toString(ranges[attr]));
		}

		double[][][] rangeProbability = new double[train.ATTR][bins][2];

		int[] counts = countNonSpamAndSpam(train.labels);

		for(int i=0;i<train.INSTANCES;i++)
		{
			for(int attr=0;attr<train.ATTR;attr++)
			{
				double val = featureMatrix[i][attr];

				for(int b=0;b<bins;b++)
					if(val>=ranges[attr][b]&&val<ranges[attr][b+1])
					{
						rangeProbability[attr][b][(int)train.labels[i][0]]++;
						break;
					}
			}
		}

		for(int attr=0;attr<train.ATTR;attr++)
		{
			for(int i=0;i<bins;i++)
			{
				rangeProbability[attr][i][0]=(rangeProbability[attr][i][0]+alpha)/(counts[0]+alpha*train.ATTR);
				rangeProbability[attr][i][1]=(rangeProbability[attr][i][1]+alpha)/(counts[1]+alpha*train.ATTR);
			}
		}

		double[] predictedLabels = histogramDistribution(ranges,rangeProbability,test,alpha,spamPrior,nonspamPrior,bins);

		return predictedLabels;
	}

	private void createPartitions(double mean,double[][] ranges,int bins, double[][] featureMatrix,
			int attr) {
		Set<Double> uniqueValues = new HashSet<Double>();
		for(int i=0;i<featureMatrix.length;i++)
			uniqueValues.add(featureMatrix[i][attr]);
		
		ArrayList<Double> sortedValues = new ArrayList<Double>(uniqueValues);
		Collections.sort(sortedValues);
		int closestMean = -1;
		for(int i=0;i<sortedValues.size()-1;i++)
			if(sortedValues.get(i)<=mean && sortedValues.get(i+1)>mean)
			{
				closestMean = i;
				break;
			}

		int leftbucket = 0;
		int rightbucket = 0;

		if(closestMean<=sortedValues.size()/2)
		{
			int x = closestMean==0? bins : (sortedValues.size()-closestMean)/(closestMean);
			rightbucket = (int) Math.ceil((bins*x)/(1+x))-1;
			leftbucket = bins-rightbucket-1;
		}
		else {
			int x = closestMean==sortedValues.size()-1? bins : (closestMean)/(sortedValues.size()-closestMean);
			leftbucket = (int) Math.ceil((bins*x)/(1+x))-1;
			rightbucket = bins-leftbucket-1;
		}
		if(rightbucket!=0)
		{
			int rightblock = (int)Math.ceil((sortedValues.size()-closestMean)/rightbucket);
			int right = bins-rightbucket;
			for(int i=closestMean;i<sortedValues.size()&&right<bins;i+=rightblock)
			{
				System.out.print(right+" ");
				ranges[attr][right++]=sortedValues.get(i);
			}
		}
		
		if(leftbucket!=0)
		{
			int leftblock = closestMean/leftbucket;
			int left = 1;
			for(int i=leftblock;i<sortedValues.size()/2&&left<closestMean;i+=leftblock)
			{
				System.out.print(left+" ");
				ranges[attr][left++]=sortedValues.get(i);
			}
		}
		System.out.println();
		if(attr==1) System.out.println(Arrays.toString(ranges[attr]));
	}

	private double[] histogramDistribution(double[][] ranges, double[][][] rangeProbability,
			NaiveBayes test, double alpha,double spamPrior,double nonspamPrior,int bins) {
		double[] predictedLabels = new double[test.INSTANCES];
		for(int i=0;i<test.INSTANCES;i++)
		{
			double l0 =1,l1=1;
			for(int attr=0;attr<test.ATTR;attr++)
			{
				double[] l = getProbabilityFromRange(ranges,test.featureMatrix[i][attr],rangeProbability,attr,bins);
				l0*=l[0];
				l1*=l[1];
			}
			predictedLabels[i]=(l0*nonspamPrior)>=(l1*spamPrior) ? 0 : 1;
		}
		return predictedLabels;
	}

	private double[] getProbabilityFromRange(double[][] ranges, double val,double[][][] rangeProbability,int attr,int bins) {
		for(int b=0;b<bins;b++)
			if((b==0 && val<ranges[attr][b+1])||(b==bins-1 && val>=ranges[attr][b])||(val>=ranges[attr][b]&&val<ranges[attr][b+1]))
				return rangeProbability[attr][b];
		return null;
	}

	private int[] countNonSpamAndSpam(double[][] labels2) {
		int[] count = new int[2];
		for(int i=0;i<labels.length;i++)
			count[(int)labels[i][0]]++;
		return count;
	}

	private double[] findMinMaxAttribute(double[][] featureMatrix, double[][] labels, int attr,double alpha) {
		double[] minmax = new double[4];
		minmax[0]=Double.MAX_VALUE;
		minmax[1]=-Double.MAX_VALUE;
		int count[]=new int[2];
		double[] resp_mean = new double[2];

		for(int i=0;i<featureMatrix.length;i++)
		{
			minmax[0]=Double.min(minmax[0], featureMatrix[i][attr]);
			minmax[1]=Double.max(minmax[1], featureMatrix[i][attr]);
			count[(int) labels[i][0]]++;
			resp_mean[(int) labels[i][0]]+=featureMatrix[i][attr];
		}
		minmax[2]=resp_mean[0]/count[0];
		minmax[3]=resp_mean[1]/count[1];
		return minmax;
	}

	private double[] performNaiveBayesUsingGaussian(
			NaiveBayes train, NaiveBayes test,
			double alpha, double spamPrior,
			double nonspamPrior) {
		double[][] probabilities = new double[2][test.INSTANCES];
		for(int label=0;label<=1;label++)
		{
			double[] mean_vector = computeMeanVector(train,label);
			double[] variances = computeVariances(train,mean_vector,label,alpha);
			probabilities[label] = getProbabilitiesFromGaussian(test,mean_vector,variances,label==0? nonspamPrior:spamPrior);
		}

		double[] predictedLabels = new double[test.INSTANCES];
		for(int i=0;i<test.INSTANCES;i++)
			predictedLabels[i] = probabilities[0][i]>=probabilities[1][i] ? 0:1;
			return predictedLabels;
	}

	private double[] getProbabilitiesFromGaussian(
			NaiveBayes test, double[] mean_vector,
			double[] variances,double prior) {
		double[] probabilities = new double[test.INSTANCES];
		for(int i=0;i<test.INSTANCES;i++)
		{
			double probability=1;
			for(int attr=0;attr<test.ATTR;attr++)
			{
				probability*=(1/(double)Math.sqrt(2*Math.PI*variances[attr]))*
						Math.exp(-0.5*
								(Math.pow((test.featureMatrix[i][attr]-mean_vector[attr]),2)/variances[attr]));
			}

			probability*=prior;
			probabilities[i]=probability;
		}
		return probabilities;
	}

	private double[] computeVariances(NaiveBayes train,
			double[] mean_vector,int label,double alpha) {
		double[] variance = new double[train.ATTR];
		int count=0;
		for(int i=0;i<train.ATTR;i++)
		{
			count=0;
			for(int row=0;row<train.INSTANCES;row++)
			{
				if(train.labels[row][0]!=label) continue;
				count++;
				variance[i]+=Math.pow((train.featureMatrix[row][i]-mean_vector[i]),2);
			}
		}

		for(int index = 0;index<train.ATTR;index++)
			variance[index]=(variance[index]+alpha)/((double)(count-1)+alpha*train.ATTR);

		return variance;
	}

	private double[] computeMeanVector(NaiveBayes train, int label) {
		double[] mean_vector = new double[train.ATTR];
		int count=0;
		for(int row = 0;row<train.INSTANCES;row++)
		{
			if(train.labels[row][0]!=label) continue;
			count++;
			int index=0;
			for(double f : train.featureMatrix[row])
			{
				mean_vector[index]+=f;
				index++;
			}
		}
		//System.out.println(label+" "+count+" "+train.INSTANCES);
		for(int index = 0;index<train.ATTR;index++)
			mean_vector[index]/=(double)(count);
		return mean_vector;
	}

	private double[] performNaiveBayesUsingBernoulli(NaiveBayes train,NaiveBayes test, double alpha, double spamPrior, double nonspamPrior) {
		double[] mean_vector = computeMeanVectors(train);
		double[][] probabilities = computeOccurenceProbabilities(train,mean_vector,alpha); 
		double[] predictedLabels = doNaiveBayes(test,mean_vector,probabilities,spamPrior,nonspamPrior);
		return predictedLabels;
	}

	private static double probabilityLabelOne(int instances,double[][] labels) {
		int count = 0;
		for(int i=0;i<instances;i++)
			if(labels[i][0]==1) count++;
		return (count/(double)instances);
	}

	private double[] doNaiveBayes(NaiveBayes test,
			double[] mean_vector, double[][] probabilities,double spamPrior,double nonspamPrior) {
		double[][] predictedLabels = new double[test.INSTANCES][2];
		for(int t=0;t<test.INSTANCES;t++)
		{
			predictedLabels[t][0]=productOfProbabilities(test.ATTR,test.featureMatrix[t],mean_vector,probabilities,0)*nonspamPrior;
			predictedLabels[t][1]=productOfProbabilities(test.ATTR,test.featureMatrix[t],mean_vector,probabilities,1)*spamPrior;
		}

		double[] labels = new double[test.INSTANCES];
		for(int i=0;i<test.INSTANCES;i++)
			labels[i]=predictedLabels[i][0]>=predictedLabels[i][1] ? 0 : 1;
			return labels;
	}

	private double productOfProbabilities(int attr,double[] instance,
			double[] mean_vector, double[][] probabilities,int label) {
		double probabiltity=1;
		for(int i=0;i<attr;i++)
			probabiltity*= ((instance[i]<=mean_vector[i]) ? probabilities[i][label] : Math.abs(1-probabilities[i][label]));
		return probabiltity;
	}

	private double[][] computeOccurenceProbabilities(
			NaiveBayes train, double[] mean_vector,double alpha) {
		double[][] probabilities = new double[train.INSTANCES][2];
		int[] count=new int[2];

		for(int i=0;i<train.INSTANCES;i++)
		{
			count[(int)train.labels[i][0]]++;
			for(int attr=0;attr<train.ATTR;attr++)
				if(train.featureMatrix[i][attr]<=mean_vector[attr])
					probabilities[attr][(int)train.labels[i][0]]++;
		}
		for(int i=0;i<train.ATTR;i++)
		{
			probabilities[i][0]=(probabilities[i][0]+alpha)/(double)(count[0]+alpha*train.ATTR);
			probabilities[i][1]=(probabilities[i][1]+alpha)/(double)(count[1]+alpha*train.ATTR);
		}
		return probabilities;
	}

	private double[] computeMeanVectors(NaiveBayes train) {
		double[] mean_vector = new double[train.ATTR];
		for(double[] row : train.featureMatrix)
		{
			int index=0;
			for(double f : row)
				mean_vector[index++]+=f;
		}

		for(int index = 0;index<train.ATTR;index++)
			mean_vector[index]/=(double)train.INSTANCES;
		return mean_vector;
	}

	private double computeAccuracy(double[] predictedLabels, double[][] labels) {

		int count = 0;
		for(int i=0;i<predictedLabels.length;i++)
			if(predictedLabels[i]==labels[i][0]) count++;

		return (count/(double)predictedLabels.length);
	}

	private static void createMatrix(double[][] featureMatrix,double[][] labels,
			Set<Integer> indexes, double[][] train, double[][] test, double[][] trainlabel, double[][] testlabel) {
		int tr=0,ts=0;
		for(int i=0;i<featureMatrix.length;i++)
		{
			if(!indexes.contains(i)) {
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
		double alpha=0.001;
		double galpha=4;
		int bins = 9;

		NaiveBayes spam = new NaiveBayes(attr,inst);
		spam.createFeatureMatrix();
		System.out.println(Arrays.toString(spam.computeSpam(attr,inst,k,alpha,galpha,bins)));
	}

}
