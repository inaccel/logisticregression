/*
Copyright © 2019 InAccel

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <ap_int.h>
#include <math.h>

#define chunk 8
#define numClassesMax 64
#define numFeaturesPlusOneMax 128
#define vectorSize 16

typedef ap_int<256> float8;
typedef ap_int<512> float16;

union {
  int asInt;
  float asFloat;
} converter1, converter2;

// This function represents a Logistc Regression HLS kernel.
// The kernel is able to train a model of up to 64 classes and 2047 features.
// Maximum bandwidth is used for the M_AXI interfaces where applicable.

extern "C" {
void Gradients_2(float8 *_labels, float16 *_data, float16 *_weights,
                 float16 *_gradients, int numClasses, int numFeatures,
                 int chunkSize) {

#pragma HLS INTERFACE m_axi port = _labels offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = _data offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = _weights offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = _gradients offset = slave bundle = gmem3
#pragma HLS INTERFACE s_axilite port = _labels bundle = control
#pragma HLS INTERFACE s_axilite port = _data bundle = control
#pragma HLS INTERFACE s_axilite port = _weights bundle = control
#pragma HLS INTERFACE s_axilite port = _gradients bundle = control
#pragma HLS INTERFACE s_axilite port = numClasses bundle = control
#pragma HLS INTERFACE s_axilite port = numFeatures bundle = control
#pragma HLS INTERFACE s_axilite port = chunkSize bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

  float16 features[chunk][numFeaturesPlusOneMax],
      weights[numClassesMax][numFeaturesPlusOneMax],
      gradients[numClassesMax][numFeaturesPlusOneMax];
  float lin[numClassesMax][chunk * vectorSize];
  float prd[chunk][numClassesMax];

// Using URAMs for features, weights and gradients buffers
#pragma HLS resource variable = features core = XPM_MEMORY uram
#pragma HLS resource variable = weights core = XPM_MEMORY uram
#pragma HLS resource variable = gradients core = XPM_MEMORY uram

// Partitioning the local arrays
#pragma HLS array_partition variable = features complete dim = 1
#pragma HLS array_partition variable = lin complete dim = 2
#pragma HLS array_partition variable = prd complete dim = 1

  // Compute the number of features iterations for float16 input data
  // (e.g. numFeatures = 31 -> (numFeatures + 1) = 16 ->  numFeaturesPlusOne =
  // 2)
  int numFeaturesPlusOne =
      (((numFeatures + 1) + (vectorSize - 1)) & (~(vectorSize - 1))) >> 4;
  // Defining a minimum of 13 classes in numClassesMin. It will be used to avoid
  // dependencies in some loops
  int numClassesMin = (13 > numClasses) ? 13 : numClasses;

  int c, i, j, k, t;

  // Reading weights and filling gradients with zeros
  for (int kj = 0, k = 0, j = 0; kj < numClasses * numFeaturesPlusOne;
       kj++, j++) {
#pragma HLS pipeline II = 1
    if (j == numFeaturesPlusOne) {
      j = 0;
      k++;
    }
    weights[k][j] = _weights[kj];
    gradients[k][j] = 0;
  }

  // Iterate over the points of the dataset each time reading a batch of 8
  // points
  for (i = 0; i < (chunkSize / chunk); i++) {
    int offset = (i * chunk) * numFeaturesPlusOne;

    // Reading the features of the dataset
    for (int cj = 0, c = 0, j = 0; cj < chunk * numFeaturesPlusOne; cj++, j++) {
#pragma HLS pipeline II = 1
      if (j == numFeaturesPlusOne) {
        j = 0;
        c++;
      }
      features[c][j] = _data[offset + cj];
    }

    // Computing the algorithm's dot product
    for (k = 0; k < numClasses; k++) {
#pragma HLS pipeline II = 1
      for (c = 0; c < chunk; c++) {
        for (t = 0; t < vectorSize; t++) {
          converter1.asInt = features[c][0].range((t + 1) * 32 - 1, t * 32);
          converter2.asInt = weights[k][0].range((t + 1) * 32 - 1, t * 32);
          lin[k][c * vectorSize + t] = converter1.asFloat * converter2.asFloat;
        }
      }
    }

    for (j = 1; j < numFeaturesPlusOne; j++) {
      for (k = 0; k < numClassesMin; k++) {
#pragma HLS pipeline II = 1
        for (c = 0; c < chunk; c++) {
          for (t = 0; t < vectorSize; t++) {
            converter1.asInt = features[c][j].range((t + 1) * 32 - 1, t * 32);
            converter2.asInt = weights[k][j].range((t + 1) * 32 - 1, t * 32);
            lin[k][c * vectorSize + t] +=
                converter1.asFloat * converter2.asFloat;
          }
        }
      }
    }

    for (k = 0; k < numClasses; k++) {
#pragma HLS pipeline II = 1
      for (c = 0; c < chunk; c++) {
        prd[c][k] =
            1.0 /
            (1.0 +
             exp(-(lin[k][c * vectorSize] + lin[k][c * vectorSize + 1] +
                   lin[k][c * vectorSize + 2] + lin[k][c * vectorSize + 3] +
                   lin[k][c * vectorSize + 4] + lin[k][c * vectorSize + 5] +
                   lin[k][c * vectorSize + 6] + lin[k][c * vectorSize + 7] +
                   lin[k][c * vectorSize + 8] + lin[k][c * vectorSize + 9] +
                   lin[k][c * vectorSize + 10] + lin[k][c * vectorSize + 11] +
                   lin[k][c * vectorSize + 12] + lin[k][c * vectorSize + 13] +
                   lin[k][c * vectorSize + 14] + lin[k][c * vectorSize + 15])));
      }
    }

    // Reading the dataset labels and update predictions
    float8 labels = _labels[i];
    for (c = 0; c < chunk; c++) {
#pragma HLS unroll
      int label = labels.range((c + 1) * 32 - 1, c * 32);
      prd[c][label] -= 1.0;
    }

    // Compute the output gradients
    for (j = 0; j < numFeaturesPlusOne; j++) {
      for (k = 0; k < numClassesMin; k++) {
#pragma HLS pipeline II = 1
        for (c = 0; c < chunk; c++) {
          for (t = 0; t < vectorSize; t++) {
            converter1.asInt = features[c][j].range((t + 1) * 32 - 1, t * 32);
            converter2.asInt = gradients[k][j].range((t + 1) * 32 - 1, t * 32);
            converter2.asFloat += prd[c][k] * converter1.asFloat;
            gradients[k][j].range((t + 1) * 32 - 1, t * 32) = converter2.asInt;
          }
        }
      }
    }
  }

  // Write back gradients
  for (int kj = 0, k = 0, j = 0; kj < numClasses * numFeaturesPlusOne;
       kj++, j++) {
#pragma HLS pipeline II = 1
    if (j == numFeaturesPlusOne) {
      j = 0;
      k++;
    }
    _gradients[kj] = gradients[k][j];
  }
}
}
