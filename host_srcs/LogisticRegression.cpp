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

#ifndef _TEST_
#define _accel_ 1
#else
#define _accel_ 0
#endif

#include <fstream>
#include <inttypes.h>
#include <iostream>
#include <malloc.h>
#include <math.h>
#include <sstream>
#include <string.h>
#include <sys/time.h>
#include <vector>

#include "inaccel/runtime-api.h"

using namespace std;

// Dataset specific options
// Change below definitions according to your input dataset
#define NUMCLASSES 26
#define NUMFEATURES 784
#define NUMEXAMPLES 124800
#define NUM_KERNELS 4

// Function to allocate an aligned memory buffer
void *INalligned_malloc(size_t size) {
  void *ptr = memalign(4096, size);
  if (!ptr) {
    printf("Error: alligned_malloc\n");
    exit(EXIT_FAILURE);
  }

  return ptr;
}

// Function to split a string on specified delimiter
vector<string> split(const string &s) {
  vector<string> elements;
  stringstream ss(s);
  string item;

  while (getline(ss, item)) {
    size_t prev = 0;
    size_t pos;

    while ((pos = item.find_first_of(" (,[])=", prev)) != std::string::npos) {
      if (pos > prev)
        elements.push_back(item.substr(prev, pos - prev));
      prev = pos + 1;
    }

    if (prev < item.length())
      elements.push_back(item.substr(prev, std::string::npos));
  }

  return elements;
}

// Reads the input dataset and sets features and labels buffers accordingly
void read_input(string filename, float *features, int *labels, int numFeatures,
                int numExamples) {
  ifstream train;
  train.open(filename.c_str());

  string line;
  int i;
  int n = 0;

  while (getline(train, line) && (n < numExamples)) {
    if (line.length()) {
      vector<string> tokens = split(line);
      features[n * (16 + numFeatures) + numFeatures] = 1.0;
      labels[n] = atoi(tokens[0].c_str());
      for (i = 0; i < numFeatures; i++) {
        features[n * (16 + numFeatures) + i] = atof(tokens[i + 1].c_str());
      }
      n++;
    }
  }

  train.close();
}

// Writes a trained model to the specified filename
void write_output(string filename, float *weights, int numClasses,
                  int numFeatures) {

  ofstream results;
  results.open(filename.c_str());

  for (int k = 0; k < numClasses; k++) {
    results << weights[k * (16 + numFeatures)];
    for (int j = 1; j < (16 + numFeatures); j++) {
      results << "," << weights[k * (16 + numFeatures) + j];
    }
    results << endl;
  }

  results.close();
}

// A simple classifier. Given an point it matches the class with the greatest
// probability
int classify(float *features, float *weights, int numClasses, int numFeatures) {
  float prob = -1.0;
  int prediction = -1;

  for (int k = 0; k < numClasses; k++) {
    float dot = weights[k * (16 + numFeatures) + numFeatures];

    for (int j = 0; j < numFeatures; j++) {
      dot += features[j] * weights[k * (16 + numFeatures) + j];
    }

    if (1.0 / (1.0 + exp(-dot)) > prob) {
      prob = 1.0 / (1.0 + exp(-dot));
      prediction = k;
    }
  }

  return prediction;
}

// A simple prediction function to evaluate the accuracy of a trained model
void predict(string filename, float *weights, int numClasses, int numFeatures) {
  cout << "    * LogisticRegression Testing *" << endl;

  float tr = 0.0;
  float fls = 0.0;
  float example[numFeatures];
  string line;
  ifstream test;

  test.open(filename.c_str());

  while (getline(test, line)) {
    if (line.length()) {
      if (line[0] != '#' && line[0] != ' ') {
        vector<string> tokens = split(line);

        int label = (int)atof(tokens[0].c_str());
        for (int j = 1; j < (1 + numFeatures); j++) {
          example[j - 1] = atof(tokens[j].c_str());
        }

        int prediction = classify(example, weights, numClasses, numFeatures);

        if (prediction == label)
          tr++;
        else
          fls++;
      }
    }
  }

  test.close();

  printf("     # accuracy:       %1.3f (%i/%i)\n", (tr / (tr + fls)), (int)tr,
         (int)(tr + fls));
  printf("     # true:           %i\n", (int)tr);
  printf("     # false:          %i\n", (int)fls);
}

// CPU implementation of Logistic Regression gradients calculation
void gradients_sw(int *labels, float *features, float *weights,
                  float *gradients, int numClasses, int numFeatures,
                  int numExamples) {
  for (int k = 0; k < numClasses; k++) {
    for (int j = 0; j < (16 + numFeatures); j++) {
      gradients[k * (16 + numFeatures) + j] = 0.0;
    }
  }

  for (int i = 0; i < numExamples; i++) {
    for (int k = 0; k < numClasses; k++) {
      float dot = weights[k * (16 + numFeatures) + numFeatures];

      for (int j = 0; j < numFeatures; j++) {
        dot += weights[k * (16 + numFeatures) + j] *
               features[i * (16 + numFeatures) + j];
      }

      float dif = 1.0 / (1.0 + exp(-dot));
      if (labels[i] == k)
        dif -= 1;

      for (int j = 0; j < (16 + numFeatures); j++) {
        gradients[k * (16 + numFeatures) + j] +=
            dif * features[i * (16 + numFeatures) + j];
      }
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    cout << "Usage: " << argv[0] << " <iterations>" << endl;
    exit(-1);
  }

  struct timeval start, end;

  float alpha = 0.3f;
  float gamma = 0.95f;
  int iter = atoi(argv[1]);

  // Set up the specifications of the model to be trained
  int numClasses = NUMCLASSES;
  int numFeatures = NUMFEATURES;
  int numExamples = NUMEXAMPLES;

  // Split the dataset among the availbale kernels
  int chunkSize = numExamples / NUM_KERNELS;

  // Allocate host buffers for lables and features of the dataset as well as
  // weights and gradients for the model to be trained and lastly velocity
  // buffer for accuracy optimization
  int *labels = (int *)INalligned_malloc(numExamples * sizeof(int));
  float *features = (float *)INalligned_malloc(
      numExamples * (16 + numFeatures) * sizeof(float));
  float *weights = (float *)INalligned_malloc(numClasses * (16 + numFeatures) *
                                              sizeof(float));
  float *gradients = (float *)INalligned_malloc(
      numClasses * (16 + numFeatures) * sizeof(float));
  float *velocity = (float *)INalligned_malloc(numClasses * (1 + numFeatures) *
                                               sizeof(float));

  // Specify train and test input files as well as output model file
  string trainFile = "data/letters_csv_train.dat";
  string testFile = "data/letters_csv_test.dat";
  string modelFile = "data/weights.out";

  // Read the input dataset
  cout << "! Reading train file..." << endl;
  read_input(trainFile, features, labels, numFeatures, numExamples);

  // Initialize model weights to zero
  for (int i = 0; i < numClasses * (16 + numFeatures); i++)
    weights[i] = 0.0;

  if (_accel_) {
    // Invoke the hardware accelerated implementation of the algorithm

    cl_engine engine[NUM_KERNELS];
    float *ffeatures[NUM_KERNELS], *fweights[NUM_KERNELS];
    float *fgradients[NUM_KERNELS], *grads[NUM_KERNELS];
    int *flabels[NUM_KERNELS];

    size_t labels_size = chunkSize * sizeof(int);
    size_t features_size = chunkSize * (numFeatures + 16) * sizeof(float);
    size_t weights_size = numClasses * (numFeatures + 16) * sizeof(float);

    // Initialize the FPGA world
    cl_world world = InAccel::create_world(0);
    // Program the FPGA device using the provided bitstream
    InAccel::create_program(world, "Gradients.xclbin");

    // Instanisate the kernels of the bitstream. Each engine holds a kernel
    // along with its command queue
    engine[0] = InAccel::create_engine(world, "Gradients_0");
    engine[1] = InAccel::create_engine(world, "Gradients_1");
    engine[2] = InAccel::create_engine(world, "Gradients_2");
    engine[3] = InAccel::create_engine(world, "Gradients_3");

    // Memcpy to each memory bank the corresponding part of the input dataset
    for (int i = 0; i < NUM_KERNELS; i++) {
      flabels[i] = (int *)InAccel::malloc(world, labels_size, i);
      InAccel::memcpy_to(world, flabels[i], 0, labels + i * chunkSize,
                         labels_size);
      ffeatures[i] = (float *)InAccel::malloc(world, features_size, i);
      InAccel::memcpy_to(world, ffeatures[i], 0,
                         features + (i * chunkSize * (16 + numFeatures)),
                         features_size);

      fweights[i] = (float *)InAccel::malloc(world, weights_size, i);

      fgradients[i] = (float *)InAccel::malloc(world, weights_size, i);
      grads[i] = (float *)INalligned_malloc(weights_size);
    }

    gettimeofday(&start, NULL);
    // Start the iterative part for the training of the algorithm
    for (int t = 0; t < iter; t++) {
      for (int i = 0; i < NUM_KERNELS; i++) {
        // Memcpy to DDR the weights of the model
        InAccel::memcpy_to(world, fweights[i], 0, weights, weights_size);

        // Set the kernel arguments
        InAccel::set_engine_arg(engine[i], 0, flabels[i]);
        InAccel::set_engine_arg(engine[i], 1, ffeatures[i]);
        InAccel::set_engine_arg(engine[i], 2, fweights[i]);
        InAccel::set_engine_arg(engine[i], 3, fgradients[i]);
        InAccel::set_engine_arg(engine[i], 4, numClasses);
        InAccel::set_engine_arg(engine[i], 5, numFeatures);
        InAccel::set_engine_arg(engine[i], 6, chunkSize);

        // Invoke the kernel execution
        InAccel::run_engine(engine[i]);
      }

      // Wait for the kernels to finish
      for (int i = 0; i < NUM_KERNELS; i++) {
        InAccel::await_engine(engine[i]);
      }

      // Get the gradients as computed by the kernels
      for (int i = 0; i < NUM_KERNELS; i++) {
        InAccel::memcpy_from(world, fgradients[i], 0, grads[i], weights_size);
      }

      // Aggregate the gradients from all kernels
      for (int j = 0; j < numClasses * (16 + numFeatures); j++) {
        gradients[j] = grads[0][j];
        for (int i = 1; i < NUM_KERNELS; i++) {
          gradients[j] += grads[i][j];
        }
      }

      // Compute the new weights of the model applying some software
      // optimizations for better model accuracy
      for (int k = 0; k < numClasses; k++) {
        for (int j = 0; j < (1 + numFeatures); j++) {
          velocity[k * (1 + numFeatures) + j] =
              gamma * velocity[k * (1 + numFeatures) + j] +
              (alpha / numExamples) * gradients[k * (16 + numFeatures) + j];
          weights[k * (16 + numFeatures) + j] -=
              velocity[k * (1 + numFeatures) + j];
        }
      }
    }

    gettimeofday(&end, NULL);

    // Free any allocated buffers for the FPGA device and release the allocated
    // kernels and command queues
    for (int i = 0; i < NUM_KERNELS; i++) {
      free(grads[i]);
      InAccel::free(world, fgradients[i]);
      InAccel::free(world, fweights[i]);
      InAccel::free(world, ffeatures[i]);
      InAccel::free(world, flabels[i]);
      InAccel::release_engine(engine[i]);
    }

    // Release the FPGA program
    InAccel::release_program(world);
    // Release the FPGA world
    InAccel::release_world(world);
  } else {
    // Invoke the software implementation of the algorithm
    gettimeofday(&start, NULL);
    for (int t = 0; t < iter; t++) {
      gradients_sw(labels, features, weights, gradients, numClasses,
                   numFeatures, numExamples);
      for (int k = 0; k < numClasses; k++) {
        for (int j = 0; j < (1 + numFeatures); j++) {
          velocity[k * (1 + numFeatures) + j] =
              gamma * velocity[k * (1 + numFeatures) + j] +
              (alpha / numExamples) * gradients[k * (16 + numFeatures) + j];
          weights[k * (16 + numFeatures) + j] -=
              velocity[k * (1 + numFeatures) + j];
        }
      }
    }
    gettimeofday(&end, NULL);
  }

  float time_us = ((end.tv_sec * 1000000) + end.tv_usec) -
                  ((start.tv_sec * 1000000) + start.tv_usec);
  float time_s = (end.tv_sec - start.tv_sec);

  cout << "! Time running Gradients Kernel: " << time_us / 1000 << " msec, "
       << time_s << " sec " << endl;

  // Compute the accuracy of the trained model on a given test dataset.
  predict(testFile, weights, numClasses, numFeatures);

  // Save the model to the specified user file
  write_output(modelFile, weights, numClasses, numFeatures);

  // Free any host allocated buffers
  free(labels);
  free(features);
  free(weights);
  free(gradients);
  free(velocity);

  return 0;
}
