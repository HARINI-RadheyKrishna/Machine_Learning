#include <immintrin.h>

#include <chrono>
#include <iostream>

#include "../utils.h"

using namespace std;
using namespace std::chrono;

void matMul(const float* A, const float* B, float* C, int A_rows, int A_cols,
            int B_rows, int B_cols) {
  if (A_cols != B_rows) {
    // The matrices can't be multiplied if A's number of columns
    // isn't equal to B's number of rows.
    throw std::invalid_argument(
        "Matrix dimensions mismatch for multiplication");
  }
  // TODO: implement matmul using 3 loops
   for (int i = 0; i < A_rows; ++i) {
      for (int j = 0; j < B_cols; ++j) {
        C[INDEX_2D_TO_1D(i, j, A_rows, B_cols)] = 0.0;
          for (int k = 0; k < A_cols; ++k) {
            C[INDEX_2D_TO_1D(i, j, A_rows, B_cols)] += A[INDEX_2D_TO_1D(i, k, A_rows, A_cols)] * B[INDEX_2D_TO_1D(k, j, A_cols, B_cols)];
          }
        }
      }
}

int main(void) {
  // modify the cachesize to match your system configuration in Google Colab
  const unsigned int cacheSize = 72 * 1024 * 1024;
  const unsigned int cacheFloatSize = cacheSize / sizeof(float);

  float* cacheFlush = allocateAlignedFloatArray(cacheSize);

  // dimension of the convolution weight
  const unsigned int weight_o = 128;
  const unsigned int weight_i = 128;
  const unsigned int weight_r = 3;
  const unsigned int weight_s = 3;

  // dimension of the convolution input
  const unsigned int input_c = 128;
  const unsigned int input_h = 128;
  const unsigned int input_w = 128;

  // Calculate the sizes needed for the arrays
  const unsigned int weightSize = weight_o * weight_i * weight_r * weight_s;
  const unsigned int inputSize = input_c * input_h * input_w;

  // Allocate memory for the arrays using the provided functions
  float* weight = allocateAlignedFloatArray(weightSize);
  float* input = allocateAlignedFloatArray(inputSize);

  // init input and weight buffers
  initializeBuffer(input, inputSize);
  initializeBuffer(weight, weightSize);

  // Declare variables to store the initial memory usage values.
  double initialVirtualMemoryUsage, initialResidentSetSize;

  // Declare variables to store any subsequent memory usage values (to be used
  // later if needed).
  double subsequentVirtualMemoryUsage, subsequentResidentSetSize;

  // Call the function to get the current memory usage and store the values in
  // the initial variables.
  getMemoryUsage(initialVirtualMemoryUsage, initialResidentSetSize);

  // TODO: calculate dimension of toeplitz output and allocate buffer 
  const unsigned int output_p = (input_h - weight_r + 1);
  const unsigned int output_q = (input_w - weight_s + 1);
  const unsigned int output_c = weight_o;
  const unsigned int outputToeplitzRows = weight_o; 
  const unsigned int outputToeplitzCols = (input_h - weight_r + 1) * (input_w - weight_s + 1);
  const unsigned int outputToeplitzSize = outputToeplitzRows*outputToeplitzCols;
  float* outputs = allocateAlignedFloatArray(outputToeplitzRows * outputToeplitzCols);

  // TODO: calculate dimension of toeplitz input and allocate buffer
  const unsigned int inputToeplitzRows = weight_i * weight_r * weight_s;
  const unsigned int inputToeplitzCols = (input_h - weight_r + 1) * (input_w - weight_s + 1);
  float* toeplitzInput = allocateAlignedFloatArray(inputToeplitzRows * inputToeplitzCols);

  // TODO: calculate dimension of toeplitz weight and allocate buffer
  const unsigned int weightToeplitzRows = weight_o;
  const unsigned int weightToeplitzCols = weight_i * weight_r * weight_s;
  float* toeplitzWeight = allocateAlignedFloatArray(weightToeplitzRows * weightToeplitzCols);
  //float* toeplitzWeightTranspose = allocateAlignedFloatArray(weightToeplitzRows * weightToeplitzCols);
  // measure memory usage for 3 buffers above only
  getMemoryUsage(subsequentVirtualMemoryUsage, subsequentResidentSetSize);
  std::cout << "Virtual Memory (KB): "
            << subsequentVirtualMemoryUsage - initialVirtualMemoryUsage
            << std::endl;

  //input matrix in row major
  int row = 0;
  for (unsigned int c = 0; c < input_c; ++c) {
      for (unsigned int r = 0; r < weight_r; ++r) {
          for (unsigned int s = 0; s < weight_s; ++s) {
              int col = 0;
              for (unsigned int i = 0; i < input_h - weight_r + 1; ++i) {
                  for (unsigned int j = 0; j < input_w - weight_s + 1; ++j) {
                      // Compute Toeplitz indices
                      //unsigned int row_idx = i * (input_w - weight_s + 1) + j;
                      //unsigned int col_idx = (c * weight_r * weight_s) + (r * weight_s) + s;
                      // Fill Toeplitz input matrix
                      toeplitzInput[INDEX_2D_TO_1D(row, col, inputToeplitzRows, inputToeplitzCols)] = 
                          input[INDEX_4D_TO_1D(0,c, i + r, j + s,0, weight_i,input_h, input_w)];
                      col++;    
                  }
              }
              row++;
          }
      }
  }
  // Compute Toeplitz weight matrix
  row = 0;
  for (unsigned int o = 0; o < weight_o; ++o) {
      int col = 0;
      for (unsigned int c = 0; c < weight_i; ++c) {
          for (unsigned int r = 0; r < weight_r; ++r) {
              for (unsigned int s = 0; s < weight_s; ++s) {
                  //unsigned int row_idx = o;
                  //unsigned int col_idx = (c * weight_r * weight_s + r * weight_s + s);
                  toeplitzWeight[INDEX_2D_TO_1D(row, col,weightToeplitzRows,weightToeplitzCols)] = 
                          weight[INDEX_4D_TO_1D(o, c, r, s, weight_o, weight_i, weight_r, weight_s)];
                  col++;
              }
          }
      }
      row++;
  }




  // Flush cache
  flushCache(cacheFlush, cacheFloatSize);
  //float* temp_output = allocateAlignedFloatArray(outputToeplitzSize);
 
  //initializeBuffer(temp_output, outputToeplitzRows * outputToeplitzCols);
  auto start = high_resolution_clock::now();
  
  //initializeBuffer(tempOutput, outputToeplitzSize);???????
  // TODO: compute matmul between toeplitz matrices, you can create a temp
  // buffer to store output_toeplitz measure runtime of this code only
  // Allocate memory for temp_output using the provided function
  //matMul( toeplitzWeight,toeplitzInput, temp_output, weightToeplitzRows, weightToeplitzCols,inputToeplitzRows, inputToeplitzCols);
  //matMul(toeplitzInput, toeplitzWeight, temp_output, inputToeplitzRows, inputToeplitzCols, weightToeplitzRows, weightToeplitzCols);
    matMul(toeplitzWeight,toeplitzInput, outputs,weightToeplitzRows, weightToeplitzCols,inputToeplitzRows,inputToeplitzCols);

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  std::cout << "Time taken by function: " << duration.count() << " milliseconds"
       << endl;
   // Measure memory usage after computation
  //double subsequentVirtualMemoryUsage, subsequentResidentSetSize;


  // TODO: reformat toeplitz y into row major
  float* output = allocateAlignedFloatArray(outputToeplitzRows * outputToeplitzCols);
  for (unsigned int i = 0; i < outputToeplitzRows; ++i) {
    for (unsigned int j = 0; j < outputToeplitzCols; ++j) {
        output[i * outputToeplitzCols + j] = outputs[j * outputToeplitzRows + i];
    }
  }
  
  printSumOfOutputs(outputs, output_c * output_p * output_q);


  // free all allocated buffers
  deallocateAlignedFloatArray(cacheFlush);
  deallocateAlignedFloatArray(input);
  deallocateAlignedFloatArray(weight);
  deallocateAlignedFloatArray(output);
  deallocateAlignedFloatArray(outputs);
  deallocateAlignedFloatArray(toeplitzInput);
  deallocateAlignedFloatArray(toeplitzWeight);
  return 0;
}
