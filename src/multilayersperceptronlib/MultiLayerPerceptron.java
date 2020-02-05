/*
 *     mlp-java, Copyright (C) 2012 Davide Gessa
 * 
 * 	This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package multilayersperceptronlib;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

public class MultiLayerPerceptron implements Cloneable
{
	protected double			fLearningRate = 0.6;
	protected Layer[]			fLayers;
	protected TransferFunction 	fTransferFunction;

	
	/**
	 * Create an mlp neuronal network
	 * 
	 * @param layers Number of neurons per layer
	 * @param learningRate Learning constant
	 * @param fun Transfer function
	 */
	public MultiLayerPerceptron(int[] layers, double learningRate, TransferFunction fun)
	{
		fLearningRate = learningRate;
		fTransferFunction = fun;
		
		fLayers = new Layer[layers.length];
		
		for(int i = 0; i < layers.length; i++)
		{			
			if(i != 0)
			{
				fLayers[i] = new Layer(layers[i], layers[i - 1]);
			}
			else
			{
				fLayers[i] = new Layer(layers[i], 0);
			}
		}
	}
	

	
	/**
	 * Run the network
	 * 
	 * @param input Input values
	 * @return Output values returned by the network
	 */
	public double[] execute(double[] input)
	{
		int i;
		int j;
		int k;
		double new_value;
		
		double output[] = new double[fLayers[fLayers.length - 1].Length]; //создание нового посл слоя
		
		// Put input
		final Layer inputLayer = fLayers[0];
		for(i = 0; i < inputLayer.Length; i++)
		{
			inputLayer.Neurons[i].Value = input[i];
		}
		
		// Execute - hiddens + output
		for(k = 1; k < fLayers.length; k++)
		{
			final Layer currentLayer = fLayers[k];
			for(i = 0; i < currentLayer.Length; i++)
			{
				new_value = 0.0;
				final Layer prevLayer = fLayers[k - 1];
				for(j = 0; j < prevLayer.Length; j++)
					new_value += currentLayer.Neurons[i].Weights[j] * prevLayer.Neurons[j].Value;
				
				new_value += currentLayer.Neurons[i].Bias;
				
				currentLayer.Neurons[i].Value = fTransferFunction.evalute(new_value);
			}
		}
		
		
		// Get output
		for(i = 0; i < fLayers[fLayers.length - 1].Length; i++)
		{
			output[i] = fLayers[fLayers.length - 1].Neurons[i].Value;
		}
		
		return output;
	}
	
	
	
	/**
	 * Backpropagation algorithm for assisted learning
	 * (Multi threads version)
	 *
	 * Unsecured and very slow convergence; use as criteria
	 * stop a norm between the previous and current errors, and a
	 * maximum number of iterations.
	 *
	 * Wikipedia:
	 * The training data is broken up into equally large batches for each
	 * of the threads. Each thread executes the forward and backward propagations.
	 * The weight and threshold deltas are summed for each of the threads.
	 * At the end of each iteration all threads must pause briefly for the
	 * weight and threshold deltas to be summed and applied to the neural network.
	 * This process continues for each iteration.
	 * 
	 * @param input Input values
	 * @param output Expected output values
	 * @param nthread Number of threads to spawn for learning
	 * @return Delta error between generated output and expected output
	 */
	public double backPropagateMultiThread(double[] input, double[] output, int nthread)
	{
		return 0.0;
	}

	
	
	/**
	 * Backpropagation algorithm for assisted learning
	 * (Single thread version)
	 *
	 * Unsecured and very slow convergence; use as criteria
	 * stop a norm between the previous and current errors, and a
	 * maximum number of iterations.
	 * 
	 * @param input Input values (scaled between 0 and 1)
	 * @param output Expected output values (scaled between 0 and 1)
	 * @return Delta error between generated output and expected output
	 */
	public double backPropagate(double[] input, double[] output)
	{
		double new_output[] = execute(input);
		double error;
		int i;
		int j;
		int k;
		
		/* doutput = correct output (output) */
		
		// Calcoliamo l'errore dell'output
		for(i = 0; i < fLayers[fLayers.length - 1].Length; i++)
		{
			error = output[i] - new_output[i];
			fLayers[fLayers.length - 1].Neurons[i].Delta = error * fTransferFunction.evaluteDerivate(new_output[i]);
		} 
	
		
		for(k = fLayers.length - 2; k >= 0; k--)
		{
			// I calculate the error of the current layer and I recalculate the deltas
			for(i = 0; i < fLayers[k].Length; i++)
			{
				error = 0.0;
				for(j = 0; j < fLayers[k + 1].Length; j++)
					error += fLayers[k + 1].Neurons[j].Delta * fLayers[k + 1].Neurons[j].Weights[i];
								
				fLayers[k].Neurons[i].Delta = error * fTransferFunction.evaluteDerivate(fLayers[k].Neurons[i].Value);
			}

			// Update the weights of the next layer
			for(i = 0; i < fLayers[k + 1].Length; i++)
			{
				for(j = 0; j < fLayers[k].Length; j++)
					fLayers[k + 1].Neurons[i].Weights[j] += fLearningRate * fLayers[k + 1].Neurons[i].Delta *
							fLayers[k].Neurons[j].Value;
				fLayers[k + 1].Neurons[i].Bias += fLearningRate * fLayers[k + 1].Neurons[i].Delta;
			}
		}	
		
		// Calcoliamo l'errore 
		error = 0.0;
		
		for(i = 0; i < output.length; i++)
		{
			error += Math.abs(new_output[i] - output[i]);
			
			//System.out.println(output[i]+" "+new_output[i]);
		}

		error = error / output.length;
		return error;
	}
	
	
	/**
	 * Save an MLP network to file
	 * 
	 * @param path Path in which to save the MLP network
	 * @return true if saved correctly
	 */
	public boolean save(String path)
	{
		try
		{
			FileOutputStream fout = new FileOutputStream(path);
			ObjectOutputStream oos = new ObjectOutputStream(fout);
			oos.writeObject(this);
			oos.close();
		}
		catch (Exception e) 
		{ 
			return false;
		}
		
		return true;
	}
	
	
	/**
	 * Upload an MLP network from file
	 * @param path Path from which to load the MLP network
	 * @return MLP network loaded from file or null
	 */
	public static MultiLayerPerceptron load(String path)
	{
		try
		{
			MultiLayerPerceptron net;
			
			FileInputStream fin = new FileInputStream(path);
			ObjectInputStream oos = new ObjectInputStream(fin);
			net = (MultiLayerPerceptron) oos.readObject();
			oos.close();
			
			return net;
		}
		catch (Exception e) 
		{ 
			return null;
		}
	}
	
	

	/**
	 * @return Learning constant
	 */
	public double getLearningRate()
	{
		return fLearningRate;
	}
	
	
	/**
	 * 
	 * @param rate
	 */
	public void	setLearningRate(double rate)
	{
		fLearningRate = rate;
	}
	
	
	/**
	 * Set up a new transfer feature
	 * 
	 * @param fun Transfer function
	 */
	public void setTransferFunction(TransferFunction fun)
	{
		fTransferFunction = fun;
	}
	
	
	
	/**
	 * @return Input layer size
	 */
	public int getInputLayerSize()
	{
		return fLayers[0].Length;
	}
	
	
	/**
	 * @return Output layer size
	 */
	public int getOutputLayerSize()
	{
		return fLayers[fLayers.length - 1].Length;
	}
}

