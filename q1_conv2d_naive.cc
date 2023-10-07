// q1_conv2d_naive.cc

#include <iostream>
#include <chrono>
#include "../utils.h"

using namespace std;
using namespace std::chrono;

int main() {
  // Modify the cachesize to match your system configuration in Google Colab
  const unsigned int cacheSize = 72 * 1024 * 1024;
  const unsigned int cacheFloatSize = cacheSize / sizeof(float);

  float* cacheFlush = allocateAlignedFloatArray(cacheSize);

  // Dimension of the convolution weight
  const unsigned int weight_o = 128;
  const unsigned int weight_i = 128;
  const unsigned int weight_r = 3;
  const unsigned int weight_s = 3;

  // Dimension of the convolution input
  const unsigned int input_c = 128;
  const unsigned int input_h = 128;
  const unsigned int input_w = 128;

  double initialVirtualMemoryUsage, initialResidentSetSize;

  // Declare variables to store any subsequent memory usage values (to be used
  // later if needed).
  double subsequentVirtualMemoryUsage, subsequentResidentSetSize;
  getMemoryUsage(initialVirtualMemoryUsage, initialResidentSetSize);

  // Calculate output dimensions
  const unsigned int output_c = weight_o;
  const unsigned int output_h = input_h - weight_r + 1;
  const unsigned int output_w = input_w - weight_s + 1;

  // Allocate memory for input, weight, and output arrays
  const unsigned int weightSize = weight_o * weight_i * weight_r * weight_s;
  const unsigned int inputSize = input_c * input_h * input_w;
  const unsigned int outputSize = output_c * output_h * output_w;

  float* weight = allocateAlignedFloatArray(weightSize);
  float* input = allocateAlignedFloatArray(inputSize);
  float* output = allocateAlignedFloatArray(outputSize);

  // Initialize input and weight buffers
  initializeBuffer(input, inputSize);
  initializeBuffer(weight, weightSize);

  // Call the function to get the current memory usage


  getMemoryUsage(subsequentVirtualMemoryUsage, subsequentResidentSetSize);
  std::cout << "Virtual Memory (KB): "
            << subsequentVirtualMemoryUsage - initialVirtualMemoryUsage
            << std::endl;
  // Flush cache
  flushCache(cacheFlush, cacheFloatSize);

  auto start = high_resolution_clock::now();

  // Implement 2D Convolution using nested for loops
  for (int oc = 0; oc < output_c; ++oc) {
    for (int oh = 0; oh < output_h; ++oh) {
      for (int ow = 0; ow < output_w; ++ow) {
        // Calculate the output index
        int outputIndex = INDEX_4D_TO_1D(0, oc, oh, ow, 1, output_c, output_h, output_w);

        // Perform the convolution
        output[outputIndex] = 0.0;
        for (int ic = 0; ic < input_c; ++ic) {
          for (int ir = 0; ir < weight_r; ++ir) {
            for (int is = 0; is < weight_s; ++is) {
              // Calculate input index
              int inputIndex = INDEX_4D_TO_1D(0, ic, oh + ir, ow + is, 1, input_c, input_h, input_w);

              // Calculate weight index
              int weightIndex = INDEX_4D_TO_1D(oc, ic, ir, is, output_c, input_c, weight_r, weight_s);

              // Perform the multiplication and accumulation
              output[outputIndex] += input[inputIndex] * weight[weightIndex];
            }
          }
        }
      }
    }
  }

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "Time taken by function: " << duration.count() << " milliseconds" << endl;

  // Print the sum of all outputs
  printSumOfOutputs(output, outputSize);

  // Free all allocated buffers
  deallocateAlignedFloatArray(input);
  deallocateAlignedFloatArray(weight);
  deallocateAlignedFloatArray(output);

  return 0;
}