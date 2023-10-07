#include <immintrin.h>

#include <chrono>
#include <iostream>

#include "../utils.h"

using namespace std;
using namespace std::chrono;

// Assume row-major for A and column-major for B
void matMulAVXOpenMPTransB(const float* A, const float* B, float* C, int A_rows,
                     int A_cols, int B_rows, int B_cols, int num_threads) {
  if (A_cols != B_rows) {
    throw std::invalid_argument(
        "Matrix dimensions mismatch for multiplication");
  }

  // need only 8 float memory vairables, but alignment requirements force requesting 256 bytes
  float* bbuffer = (float*) aligned_alloc(256, 64*sizeof(float));
  float* abuffer = (float*) aligned_alloc(256, 64*sizeof(float));
  float* cbuffer = (float*) aligned_alloc(256, 64*sizeof(float));
  int glbcnt = 0;
  // TODO: implement AVX version of matMulTransB
  if(A_cols%8 != 0){
    #pragma omp parallel for num_threads(num_threads)
    for(int i=0; i<A_rows; i++){
        for(int j=0; j<B_cols; j++){
          for(int k=0; k<((B_rows/8)*8); k+=8){
            __m256 temp_i = _mm256_loadu_ps(&(A[(i*A_cols) + k]));
            __m256 temp_w = _mm256_loadu_ps(&(B[(j*B_rows) + k]));
            __m256 temp_m = _mm256_mul_ps(temp_i, temp_w);
            float temp_reg[8];
            _mm256_storeu_ps(temp_reg, temp_m);
            for(int n=0; n<8; n++){
              C[(i*B_cols) + j] += temp_reg[n];
            }
          }
        }
     }
    #pragma omp parallel for num_threads(num_threads)
    for(int i=0; i<A_rows; i++){
      for(int j=0; j<B_cols; j++){
        for(int k=((A_cols/8)*8); k<B_rows; k++){
          int A_i = (i*A_cols) + k;
          int B_i = (j*B_rows) + k;
          int C_i = (i*B_cols) + j;
          C[C_i] += A[A_i] * B[B_i];
        }
      }
    }
  }
  else{
      #pragma omp parallel for num_threads(num_threads)
       for(int i=0; i<A_rows; i++){
        for(int j=0; j<B_cols; j++){
          for(int k=0; k<B_rows; k+=8){
            __m256 temp_i = _mm256_loadu_ps(&(A[(i*A_cols) + k]));
            __m256 temp_w = _mm256_loadu_ps(&(B[(j*B_rows) + k]));
            __m256 temp_m = _mm256_mul_ps(temp_i, temp_w);
            float temp_reg[8];
            _mm256_storeu_ps(temp_reg, temp_m);
            for(int n=0; n<8; n++){
              C[(i*B_cols) + j] += temp_reg[n];
            }
          }
        }
       }
      }
  deallocateAlignedFloatArray(abuffer);
  deallocateAlignedFloatArray(bbuffer);
  deallocateAlignedFloatArray(cbuffer);
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
  const unsigned int output_c = weight_o;
  const unsigned int output_p = input_h - weight_r + 1;
  const unsigned int output_q = input_w - weight_s + 1;
  const unsigned int output_row = weight_o;
  const unsigned int output_col = output_p * output_q;
  const unsigned int opSize = output_row * output_col;
  float* output = allocateAlignedFloatArray(opSize);

  // TODO: calculate dimension of toeplitz input and allocate buffer
  const unsigned int newinput_h = weight_r*weight_s*weight_i;
  const unsigned int newinput_w = output_p*output_q;
  const unsigned int newipSize = newinput_h * newinput_w;
  float* topinput = allocateAlignedFloatArray(newipSize);

  // TODO: calculate dimension of toeplitz weight and allocate buffer
  const unsigned int newweight_r = weight_o;
  const unsigned int newweight_s = newinput_h;
  const unsigned int newweightSize = newweight_r * newweight_s;
  float* topweight = allocateAlignedFloatArray(newweightSize);

  // measure memory usage for 3 buffers above only
  getMemoryUsage(subsequentVirtualMemoryUsage, subsequentResidentSetSize);
  std::cout << "Virtual Memory (KB): "
            << subsequentVirtualMemoryUsage - initialVirtualMemoryUsage
            << std::endl;

  // TODO: convert matrix input & weight into toeplitz matrices
  // toeplitz input is stored in row major and toeplitz weight is stored in
  // column major order
  // toeplitz input matrix in row-major

  for (int inc = 0; inc < weight_i; inc++) {
    for (int r = 0; r < weight_r; r++) {
      for (int s = 0; s < weight_s; s++) {
        for (int p = 0; p < output_p; p++) {
          for (int q = 0; q < output_q; q++) {
            topinput[(p * weight_r * weight_s * weight_i * output_q) + (q * weight_r * weight_s * weight_i) + (inc * weight_r * weight_s) + (r * weight_s) + s] =
                input[(p * input_w) + q + (input_h * input_w * inc) + (input_w * r) + s];
          }
        }
      }
    }
  }

  // weight matrix in column-major
  for(int i=0; i<newweightSize; i++){
    topweight[i] = weight[i];
  }
  // Flush cache
  flushCache(cacheFlush, cacheFloatSize);

  auto start = high_resolution_clock::now();
  // TODO: compute matmul between toeplitz matrices, you can create a temp
  // buffer to store output_toeplitz measure runtime of this code only
  // Allocate memory for temp_output using the provided function
  float *temp_output = allocateAlignedFloatArray(opSize);

  matMulAVXOpenMPTransB(topinput, topweight, temp_output, (output_p * output_q), (weight_i * weight_r * weight_s), (weight_i * weight_r * weight_s), weight_o, 2);

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(stop - start);
  cout << "Time taken by 2 threaded core: " << duration.count() << " milliseconds" << endl;

  start = high_resolution_clock::now();
  float* temp_4output = allocateAlignedFloatArray(opSize);    
  matMulAVXOpenMPTransB(topinput, topweight, temp_4output, (output_p * output_q), (weight_i * weight_r * weight_s), (weight_i * weight_r * weight_s), weight_o, 4);


  stop = high_resolution_clock::now();
  duration = duration_cast<milliseconds>(stop - start);
  cout << "Time taken by 4 threaded core: " << duration.count() << " milliseconds"
       << endl;
  
  start = high_resolution_clock::now();
  float* temp_8output = allocateAlignedFloatArray(opSize);
  matMulAVXOpenMPTransB(topinput, topweight, temp_8output, (output_p * output_q), (weight_i * weight_r * weight_s), (weight_i * weight_r * weight_s), weight_o, 8);


  stop = high_resolution_clock::now();
  duration = duration_cast<milliseconds>(stop - start);
  cout << "Time taken by 8 threaded core: " << duration.count() << " milliseconds"
       << endl;

  //const unsigned int weight_o = 128;
  // TODO: reformat toeplitz y into row major
  for(int i=0; i<weight_o; i++){
      for(int j=0; j<(output_p*output_q); j++){
        output[(output_p * output_q * i) + j] = temp_output[i + (weight_o*j)];
      }
  }
  // TODO: reformat toeplitz y into row major


  printSumOfOutputs(output, output_c * output_p * output_q);

  // free all allocated buffers
  deallocateAlignedFloatArray(input);
  deallocateAlignedFloatArray(weight);
  deallocateAlignedFloatArray(output);
  deallocateAlignedFloatArray(temp_output);
  return 0;
}