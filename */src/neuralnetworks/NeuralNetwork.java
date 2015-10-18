package neuralnetworks;

import java.io.PrintWriter;
import java.math.BigDecimal;
import java.util.Arrays;

public class NeuralNetwork {
	Layer in;
	Layer hidden;
	Layer out;
	final double THRESHOLD;
	int[][] validOutput;

	public NeuralNetwork(double t)
	{
		in=new Layer(8,3);
		hidden=new Layer(3,8);
		out=new Layer(8,0);
		in.setWeights();
		hidden.setWeights();
		out.setWeights();
		THRESHOLD=t;
		createValidOutput();
	}

	private void createValidOutput() {
		validOutput = new int[8][3];
		validOutput[0]=new int[]{1,0,0};
		validOutput[1]=new int[]{0,1,1};
		validOutput[2]=new int[]{0,1,0};
		validOutput[3]=new int[]{1,1,1};
		validOutput[4]=new int[]{0,0,0};
		validOutput[5]=new int[]{0,0,1};
		validOutput[6]=new int[]{1,0,1};
		validOutput[7]=new int[]{1,1,0};
	}

	public void compute(double learnRate)
	{
		double diff=0,prevdiff=0;
		int index=0;
		do{
			System.out.println("Iteration #"+index++);
			prevdiff=diff;
			diff=0;
			System.out.println("Start");
			for(int i=0;i<8;i++)
			{
				in.reset();
				hidden.reset();
				out.reset();

				// front propogation
				in.input[i]=1;
				in.output= in.input;

				hidden.linearRegressForInput(in);
				hidden.outputSigmoid();

				out.linearRegressForInput(hidden);
				out.outputSigmoid();
				//System.out.println(Arrays.toString(out.output));
				// back propogation
				double[] outputError = out.getLogisticError(in.input);
				double[] hiddenUnitError = hidden.getRegressionError(outputError);

				for(int node=0;node<8;node++)
					for(int hid=0;hid<3;hid++)
					{
						in.weights[node][hid]+=learnRate*hiddenUnitError[hid]*in.output[node];
						hidden.weights[hid][node]+=learnRate*outputError[node]*hidden.output[hid];
					}
				for(int hid=0;hid<3;hid++)
					in.bias[hid]+=learnRate*hiddenUnitError[hid];
				for(int node=0;node<8;node++)
					hidden.bias[node]+=learnRate*outputError[node];

				diff+=computeError(out.output,in.input);
				System.out.print(Arrays.toString(round(in.input))+" --> ");
				System.out.print(Arrays.toString(round(hidden.output))+" --> ");
				System.out.println(Arrays.toString(round(out.output)));
			}
		}
		while(Math.abs(diff-prevdiff)>THRESHOLD);
		//pw.println(learnRate+"\t"+diff);
	}

	private double[] round(double[] output) {
		double[] output1 = new double[output.length];
		for(int i=0;i<output.length;i++)
			output1[i]=(new BigDecimal(output[i]).setScale(2, BigDecimal.ROUND_HALF_UP)).doubleValue();
		return output1;
	}

	public double computeError(double[] outputs,double[] input)
	{
		double min=0;
		for(int i=0;i<outputs.length;i++)
			min+=Math.abs(outputs[i]-input[i]);
		return min;
	}

	public static void main(String args[]) throws Exception
	{
		NeuralNetwork n = new NeuralNetwork(0.00007);
		n.compute(0.2);
	}
}
