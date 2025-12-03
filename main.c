#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

typedef struct {
  uint8_t pixels[28 * 28];
} Image;

typedef struct {
  int rows;
  int cols;
  float *data;
} Matrix;

uint32_t fbytes(uint32_t val) {
  return ((val >> 24) & 0x000000FF) | ((val >> 8) & 0x0000FF00) |
         ((val << 8) & 0x00FF0000) | ((val << 24) & 0xFF000000);
}

Matrix *matrix_create(int rows, int cols) {
  Matrix *mat = (Matrix *)malloc(sizeof(Matrix));
  mat->rows = rows;
  mat->cols = cols;
  mat->data = (float *)calloc(rows * cols, sizeof(float));
  return mat;
}

Matrix *matrix_multiply(Matrix *x, Matrix *y) {
  if (x->cols != y->rows) {
    perror("Matrices cannot be multiplied, dimension error");
    return NULL;
  }
  Matrix *mat = matrix_create(x->rows, y->cols);

  for (int i = 0; i < mat->rows; ++i) {
    for (int j = 0; j < mat->cols; ++j) {
      float sum = 0;
      for (int k = 0; k < x->cols; ++k) {
        float a = x->data[i * x->cols + k];
        float b = y->data[k * y->cols + j];
        sum += a * b;
      }
      mat->data[i * mat->cols + j] = sum;
    }
  }

  return mat;
}

Matrix *matrix_randomize(Matrix *mat) {
  for (int i = 0; i < mat->rows; ++i) {
    for (int j = 0; j < mat->cols; ++j) {
      mat->data[i * mat->cols + j] = ((float)rand() / RAND_MAX) - 0.5;
    }
  }
  return mat;
}

Matrix *matrix_relu(Matrix *mat) {
  for (int i = 0; i < mat->rows; ++i) {
    for (int j = 0; j < mat->cols; ++j) {
      if (mat->data[i * mat->cols + j] < 0) {
        mat->data[i * mat->cols + j] = 0;
      }
    }
  }
  return mat;
}

Matrix *matrix_softmax(Matrix *mat) {
  int num_of_elements = mat->rows * mat->cols;
  float sum = 0.0f;
  float max_value = -INFINITY;
  float epsilon = 1e-7f;

  for (int i = 0; i < num_of_elements; i++) {
    if (mat->data[i] > max_value) {
      max_value = mat->data[i];
    }
  }

  for (int i = 0; i < num_of_elements; ++i) {
    float exp = expf(mat->data[i] - max_value);
    mat->data[i] = exp;
    sum += exp;
  }
  for (int i = 0; i < num_of_elements; ++i) {
    mat->data[i] /= (sum + epsilon);
  }

  return mat;
}

Matrix *matrix_flatten(Image *img) {
  Matrix *mat = matrix_create(1, 784);

  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      mat->data[i * 28 + j] = img->pixels[i * 28 + j] / 255.0;
    }
  }
  return mat;
}

Matrix *matrix_one_hot(int label, int size) {
  Matrix *mat = matrix_create(1, size);
  mat->data[label] = 1.0f;
  return mat;
}

void matrix_print(Matrix *mat) {
  int rows = mat->rows;
  int cols = mat->cols;

  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      printf(" %f", mat->data[i * cols + j]);
    }
    printf("\n");
  }
}

int arg_max(Matrix *mat) {
  int max_index = 0;
  int mat_size = mat->rows * mat->cols;
  for (int i = 0; i < mat_size; ++i) {
    if (mat->data[i] > mat->data[max_index]) {
      max_index = i;
    }
  }
  return max_index;
}

void print_image_labels(Image *image, uint8_t label) {
  printf("Label: %d\n", label);
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      if (image->pixels[i * 28 + j] > 10) {
        printf("#");
      } else {
        printf(".");
      }
    }
    printf("\n");
  }
}

void print_image_matrix(Matrix *mat) {
  for (int i = 0; i < 28; ++i) {
    for (int j = 0; j < 28; ++j) {
      if (mat->data[i * 28 + j] > 0) {
        printf("#");
      } else {
        printf(".");
      }
    }
    printf("\n");
  }
}

int main() {

  /* ---------------DATA LOADING---------------- */
  char *folder = "/home/syien/Documents/dev/nn/MNIST_ORG/";
  uint32_t image_header_buffer[4], label_header_buffer[2],
      test_header_buffer[4], test_label_buffer[2];
  uint32_t magic_image, num_train_images, num_test_images, rows, cols,
      magic_labels, train_label_count, test_label_count;

  char input_images_file[256];
  sprintf(input_images_file, "%strain-images.idx3-ubyte", folder);
  FILE *fp_train_images = fopen(input_images_file, "rb");
  if (fp_train_images == NULL) {
    perror("Error: could not open image file");
  }

  char input_labels_file[256];
  sprintf(input_labels_file, "%strain-labels.idx1-ubyte", folder);
  FILE *fp_train_labels = fopen(input_labels_file, "rb");
  if (fp_train_labels == NULL) {
    perror("Error: could not open labels file");
  }

  char test_images_file[256];
  sprintf(test_images_file, "%st10k-images.idx3-ubyte", folder);
  FILE *fp_test_images = fopen(test_images_file, "rb");
  if (fp_test_images == NULL) {
    perror("Error: could not open image file");
  }

  char test_labels_file[256];
  sprintf(test_labels_file, "%st10k-labels.idx1-ubyte", folder);
  FILE *fp_test_labels = fopen(test_labels_file, "rb");
  if (fp_test_labels == NULL) {
    perror("Error: could not open labels file");
  }

  fread(image_header_buffer, sizeof(uint32_t), 4, fp_train_images);
  fread(label_header_buffer, sizeof(uint32_t), 2, fp_train_labels);

  fread(test_header_buffer, sizeof(uint32_t), 4, fp_test_images);
  fread(test_label_buffer, sizeof(uint32_t), 2, fp_test_labels);

  magic_image = image_header_buffer[0];
  magic_image = fbytes(magic_image);

  num_train_images = image_header_buffer[1];
  num_train_images = fbytes(num_train_images);

  num_test_images = test_header_buffer[1];
  num_test_images = fbytes(num_test_images);

  rows = image_header_buffer[2];
  rows = fbytes(rows);

  cols = image_header_buffer[3];
  cols = fbytes(cols);

  magic_labels = label_header_buffer[0];
  magic_labels = fbytes(magic_labels);

  train_label_count = label_header_buffer[1];
  train_label_count = fbytes(train_label_count);

  test_label_count = test_label_buffer[1];
  test_label_count = fbytes(test_label_count);

  Image *train_images = (Image *)malloc(sizeof(Image) * num_train_images);
  uint8_t *train_labels =
      (uint8_t *)malloc(sizeof(uint8_t) * train_label_count);

  Image *test_images = (Image *)malloc(sizeof(Image) * num_test_images);
  uint8_t *test_labels = (uint8_t *)malloc(sizeof(uint8_t) * test_label_count);

  fread(train_images, sizeof(Image), num_train_images, fp_train_images);
  fread(train_labels, sizeof(uint8_t), train_label_count, fp_train_labels);

  fread(test_images, sizeof(Image), num_test_images, fp_test_images);
  fread(test_labels, sizeof(uint8_t), test_label_count, fp_test_labels);

  // print_image_labels(&train_images[0], train_labels[0]);
  // print_image_labels(&test_images[0], test_labels[0]);

  Matrix *weights = matrix_create(784, 10);
  Matrix *randomized_weights = matrix_randomize(weights);

  for (int epoch = 0; epoch < 100; ++epoch) {
    for (int i = 0; i < 1000; ++i) {
      Matrix *input = matrix_flatten(&train_images[i]);
      Matrix *target_matrix = matrix_one_hot(train_labels[i], 10);
      Matrix *output = matrix_multiply(input, weights);
      Matrix *predictions = matrix_softmax(output);

      Matrix *error_matrix = matrix_create(1, 10);
      for (int i = 0; i < 10; ++i) {
        error_matrix->data[i] = predictions->data[i] - target_matrix->data[i];
      }
      for (int i = 0; i < 784; ++i) {
        for (int j = 0; j < 10; ++j) {
          float gradient = input->data[i] * error_matrix->data[j];
          weights->data[i * 10 + j] -= 0.01 * gradient;
        }
      }
      free(input);
      free(target_matrix);
      free(output);
    }

    int correct = 0;
    for (int i = 0; i < 100; ++i) {
      Matrix *test_input = matrix_flatten(&test_images[i]);
      Matrix *target_matrix = matrix_one_hot(test_labels[i], 10);
      Matrix *test_output = matrix_multiply(test_input, weights);
      Matrix *predictions = matrix_softmax(test_output);
      int max_prob_index = arg_max(predictions);
      if (max_prob_index == test_labels[i]) {
        correct++;
      }
      free(test_input);
      free(target_matrix);
      free(test_output);
    }
    printf("Epoch %d Accuracy: %.2f%%\n", epoch,
           (float)correct / num_test_images * 100);
  }

  free(train_images);
  free(train_labels);
  free(test_images);
  free(test_labels);
  free(weights);
  fclose(fp_train_images);
  fclose(fp_train_labels);
  return 0;
}
