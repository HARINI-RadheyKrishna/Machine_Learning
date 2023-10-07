#include <immintrin.h>

#include <chrono>
#include <iostream>
#include "../utils.h"
using namespace std;
using namespace std::chrono;

void matMulTransB(const float* A, const float* B, float* C, int A_rows, int A_cols,
            int B_rows, int B_cols) {
  if (A_cols != B_rows) {
    // The matrices can't be multiplied if A's number of columns
    // isn't equal to B's number of rows.
    throw std::invalid_argument(
        "Matrix dimensions mismatch for multiplication");
  }

  int col = 0;
  // TODO: implement matmul using 3 loops
  for (int i = 0; i < A_rows; i++){
    for (int k = 0; k < B_cols; k++){

      for (int j = 0; j < A_cols; j++){

        C[col] += 
        A[INDEX_2D_TO_1D(j,i,0,A_rows)] * B[INDEX_2D_TO_1D(k,j,0,B_rows)];
      }

      col++;
    }

  }


}

int main(void) {
  // modify the cachesize to match your system configuration in Google Colab
  const unsigned int cacheSize = 72 * 1024 * 1024;
  const unsigned int cachesizefloat = cacheSize / sizeof(float);

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
  const unsigned int output_c = weight_o;
  const unsigned int output_p = input_h - weight_r + 1;
  const unsigned int output_q = input_w - weight_s + 1;
  const unsigned int output_row = weight_o;
  const unsigned int output_col = output_p * output_q;
  const unsigned int OutputSize = output_row * output_col;
  float* output = allocateAlignedFloatArray(OutputSize);

  // TODO: calculate dimension of toeplitz input and allocate buffer
  const unsigned int newinput_h = weight_r*weight_s*weight_i;
  const unsigned int newinput_w = output_p*output_q;
  const unsigned int newipSize = newinput_h * newinput_w;
  float* TopleitzInput = allocateAlignedFloatArray(newipSize);

  // TODO: calculate dimension of toeplitz weight and allocate buffer
  const unsigned int newweight_r = weight_o;
  const unsigned int newweight_s = newinput_h;
  const unsigned int newweightSize = newweight_r * newweight_s;
  float* TopleitzWeight = allocateAlignedFloatArray(newweightSize);

  // measure memory usage for 3 buffers above only
  getMemoryUsage(subsequentVirtualMemoryUsage, subsequentResidentSetSize);
  std::cout << "Virtual Memory (KB): "
            << subsequentVirtualMemoryUsage - initialVirtualMemoryUsage
            << std::endl;

  // TODO: convert matrix input & weight into toeplitz matrices
  // toeplitz input is stored in row major and toeplitz weight is stored in column
  // major order
  // input matrix in row-major
  int row = 0;

  int col = 0;
  for (int p = 0; p < output_p; p++){
    for (int q = 0; q < output_q; q++){
      for (int inc = 0; inc < weight_i; inc++){
        for (int r = 0; r < weight_r; r++) {
          for (int s = 0; s < weight_s; s++){
            TopleitzInput[row] = input[INDEX_4D_TO_1D(0,inc, p + r, q + s, 0, weight_i, input_h, input_w)];
            row++;
          }
        }

      }


    }


  }
  
  // weight matrix in column-major
  col = 0;
  for (int inc = 0; inc < weight_i; inc++){
    for (int r = 0; r < weight_r; r++) {
      for (int s = 0; s < weight_s; s++){
        for (int m = 0; m < newweight_r; m++){
          TopleitzWeight[col] = weight[INDEX_4D_TO_1D(m,inc, r, s, newweight_r, weight_i, weight_r, weight_s)];
          col++;
              }
            }

      
       }

  }

  // Flush cache
  flushCache(cacheFlush, cachesizefloat);

  auto start = high_resolution_clock::now();
  // TODO: compute matmul between toeplitz matrices, you can create a temp
  // buffer to store output_toeplitz measure runtime of this code only
  // Allocate memory for temp_output using the provided function
  matMulTransB(TopleitzWeight, TopleitzInput, output, newweight_r, newweight_s, newinput_h, newinput_w);

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "Time taken by function: " << duration.count() << " milliseconds"
       << endl;

  // TODO: reformat toeplitz y into row major


  printSumOfOutputs(output, output_c * output_p * output_q);

  // free all allocated buffers
  deallocateAlignedFloatArray(input);
  deallocateAlignedFloatArray(weight);
  deallocateAlignedFloatArray(output);
  return 0;
}
