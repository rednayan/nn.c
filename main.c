#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>

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

Matrix *matrix_transpose(Matrix *mat) {
  Matrix *transposed_matrix = matrix_create(mat->cols, mat->rows);
  for (int i = 0; i < mat->rows; ++i) {
    for (int j = 0; j < mat->cols; ++j) {
      int src_index = i * mat->cols + j;
      int dst_index = j * transposed_matrix->cols + i;
      transposed_matrix->data[dst_index] = mat->data[src_index];
    }
  }
  return transposed_matrix;
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

void matrix_free(Matrix *m) {
  if (m != NULL) {
    if (m->data != NULL) {
      free(m->data);
    }
    free(m);
  }
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

  srand(time(0));

  char *folder = "/home/syien/Documents/dev/nn/MNIST_ORG/";
  uint32_t image_header_buffer[4], label_header_buffer[2],
      test_header_buffer[4], test_label_buffer[2];
  int magic_image, num_train_images, num_total_test_images, rows, cols,
      magic_labels, train_label_count, test_total_label_count;

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

  num_total_test_images = test_header_buffer[1];
  num_total_test_images = fbytes(num_total_test_images);
  int num_test_images = (int)(0.999f * num_total_test_images);

  int num_validation_images = (int)(0.001f * num_total_test_images);

  rows = image_header_buffer[2];
  rows = fbytes(rows);

  cols = image_header_buffer[3];
  cols = fbytes(cols);

  magic_labels = label_header_buffer[0];
  magic_labels = fbytes(magic_labels);

  train_label_count = label_header_buffer[1];
  train_label_count = fbytes(train_label_count);

  test_total_label_count = test_label_buffer[1];
  test_total_label_count = fbytes(test_total_label_count);
  int test_label_count = (int)(0.999f * test_total_label_count);

  int validation_label_count = (int)(0.001f * test_total_label_count);

  Image *train_images = (Image *)malloc(sizeof(Image) * num_train_images);
  uint8_t *train_labels =
      (uint8_t *)malloc(sizeof(uint8_t) * train_label_count);

  Image *test_images = (Image *)malloc(sizeof(Image) * num_test_images);
  uint8_t *test_labels = (uint8_t *)malloc(sizeof(uint8_t) * test_label_count);

  Image *validation_images =
      (Image *)malloc(sizeof(Image) * num_validation_images);
  uint8_t *validation_labels =
      (uint8_t *)malloc(sizeof(uint8_t) * test_label_count);

  fread(train_images, sizeof(Image), num_train_images, fp_train_images);
  fread(train_labels, sizeof(uint8_t), train_label_count, fp_train_labels);

  fread(test_images, sizeof(Image), num_test_images, fp_test_images);
  fread(test_labels, sizeof(uint8_t), test_label_count, fp_test_labels);

  fread(validation_images, sizeof(Image), num_validation_images,
        fp_test_images);
  fread(validation_labels, sizeof(uint8_t), validation_label_count,
        fp_test_labels);

  Matrix *W1 = matrix_randomize(matrix_create(784, 128));
  Matrix *W2 = matrix_randomize(matrix_create(128, 10));

  for (int epoch = 0; epoch < 10; ++epoch) {
    for (int i = 0; i < num_train_images; ++i) {
      Matrix *X = matrix_flatten(&train_images[i]);
      Matrix *X_raw = matrix_multiply(X, W1);
      Matrix *X1 = matrix_relu(X_raw);
      Matrix *X1W2 = matrix_multiply(X1, W2);
      Matrix *output = matrix_softmax(X1W2);

      Matrix *error_matrix = matrix_create(1, 10);
      Matrix *target_matrix = matrix_one_hot(train_labels[i], 10);
      for (int j = 0; j < 10; j++) {
        error_matrix->data[j] = output->data[j] - target_matrix->data[j];
      }
      Matrix *transpose_X1 = matrix_transpose(X1);
      Matrix *grad2 = matrix_multiply(transpose_X1, error_matrix);
      Matrix *transpose_W2 = matrix_transpose(W2);
      Matrix *error_matrix_2 = matrix_multiply(error_matrix, transpose_W2);

      for (int k = 0; k < 128; ++k) {
        if (X1->data[k] == 0) {
          error_matrix_2->data[k] = 0;
        }
      }

      Matrix *transpose_X = matrix_transpose(X);
      Matrix *grad1 = matrix_multiply(transpose_X, error_matrix_2);

      for (int l = 0; l < W2->cols * W2->rows; ++l) {
        W2->data[l] -= 0.01 * grad2->data[l];
      }

      for (int m = 0; m < W1->cols * W1->rows; ++m) {
        W1->data[m] -= 0.01 * grad1->data[m];
      }

      matrix_free(transpose_X1);
      matrix_free(transpose_X);
      matrix_free(transpose_W2);
      matrix_free(X);
      matrix_free(X_raw);
      matrix_free(X1W2);
      matrix_free(error_matrix);
      matrix_free(error_matrix_2);
      matrix_free(target_matrix);
      matrix_free(grad1);
      matrix_free(grad2);
    }

    int correct = 0;
    for (int n = 0; n < num_test_images; ++n) {
      Matrix *input_test = matrix_flatten(&test_images[n]);
      Matrix *input_x = matrix_multiply(input_test, W1);
      Matrix *input_x_relu = matrix_relu(input_x);
      Matrix *input_x1 = matrix_multiply(input_x_relu, W2);
      Matrix *pred_output = matrix_softmax(input_x1);

      int pred_label = arg_max(pred_output);
      if (pred_label == test_labels[n]) {
        correct++;
      }
      matrix_free(input_test);
      matrix_free(input_x_relu);
      matrix_free(pred_output);
    }
    printf("Epoch %d Accuracy: %.2f%%\n", epoch,
           ((float)correct / num_test_images) * 100);
  }

  for (int i = 0; i < num_validation_images; ++i) {
    Matrix *input_validation = matrix_flatten(&validation_images[i]);
    Matrix *input_val = matrix_multiply(input_validation, W1);
    Matrix *input_val_relu = matrix_relu(input_val);
    Matrix *input_val_2 = matrix_multiply(input_val_relu, W2);
    Matrix *pred_out_val = matrix_softmax(input_val_2);

    int prediction_output = arg_max(pred_out_val);
    printf("prediction: %d\n", prediction_output);
    print_image_labels(&validation_images[i], validation_labels[i]);

    matrix_free(input_validation);
    matrix_free(input_val_relu);
    matrix_free(pred_out_val);
  }

  free(train_images);
  free(train_labels);
  free(test_images);
  free(test_labels);
  matrix_free(W1);
  matrix_free(W2);
  fclose(fp_train_images);
  fclose(fp_train_labels);
  return 0;
}
