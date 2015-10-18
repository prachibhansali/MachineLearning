package neuralnetworks;

import java.util.Random;

public class Layer {
	final int n;
	final int connectedTo;
	double[] input;
	double[] output;
	double[][] weights;
	double[] bias;

	public Layer(int n,int c)
	{
		this.n=n;
		connectedTo=c;
		input = new double[n];
		output = new double[n];
	}
	
	public void setWeights()
	{
		weights = new double[n][connectedTo];
		bias=new double[connectedTo];
		for(int i=0;i<connectedTo;i++)
			bias[i]=0.1;
		assignRandomValues();
	}

	private void assignRandomValues() {
		Random r = new Random();
		for(int i=0;i<n;i++)
			for(int j=0;j<connectedTo;j++)
				weights[i][j]=r.nextDouble()-0.5;
	}

	public double applySigmoid(double val)
	{
		return 1/(double)(1+Math.exp(-val));
	}

	public void linearRegressForInput(Layer in) {
		for(int i=0;i<n;i++)
		{
			double sum = 0;
			for(int node=0;node<in.n;node++)
				sum+=in.output[node]*in.weights[node][i];
			input[i]=sum+in.bias[i];
		}
	}

	public void outputSigmoid() {
		for(int i=0;i<n;i++)
			output[i]=applySigmoid(input[i]);
	}

	public double[] getLogisticError(double[] targets) {
		double[] error = new double[n];
		for(int i=0;i<n;i++)
		{
			error[i]=output[i]*(1-output[i])*(targets[i]-output[i]);
		}
		return error;
	}

	public double[] getRegressionError(double[] outputError) {
		double[] error = new double[n];
		for(int i=0;i<n;i++)
		{
			error[i]=output[i]*(1-output[i]);
			double sum=0;
			for(int o=0;o<outputError.length;o++)
				sum+=weights[i][o]*outputError[o];
			error[i]*=sum;
		}
		return error;
	}

	public void reset() {
		input = new double[n];
		output = new double[n];
	}
}
