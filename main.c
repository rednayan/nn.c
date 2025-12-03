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
  uint32_t image_header_buffer[4], label_header_buffer[2];
  uint32_t magic_image, num_images, rows, cols, magic_labels, label_count;

  char input_images[256];
  sprintf(input_images, "%strain-images.idx3-ubyte", folder);
  FILE *fp_train_images = fopen(input_images, "rb");
  if (fp_train_images == NULL) {
    perror("Error: could not open image file");
  }

  char input_labels[256];
  sprintf(input_labels, "%strain-labels.idx1-ubyte", folder);
  FILE *fp_train_labels = fopen(input_labels, "rb");
  if (fp_train_labels == NULL) {
    perror("Error: could not open labels file");
  }

  fread(image_header_buffer, sizeof(uint32_t), 4, fp_train_images);
  fread(label_header_buffer, sizeof(uint32_t), 2, fp_train_labels);

  magic_image = image_header_buffer[0];
  magic_image = fbytes(magic_image);

  num_images = image_header_buffer[1];
  num_images = fbytes(num_images);

  rows = image_header_buffer[2];
  rows = fbytes(rows);

  cols = image_header_buffer[3];
  cols = fbytes(cols);

  magic_labels = label_header_buffer[0];
  magic_labels = fbytes(magic_labels);

  label_count = label_header_buffer[1];
  label_count = fbytes(label_count);

  Image *images = (Image *)malloc(sizeof(Image) * num_images);
  uint8_t *labels = (uint8_t *)malloc(sizeof(uint8_t) * label_count);

  fread(images, sizeof(Image), num_images, fp_train_images);
  fread(labels, sizeof(uint8_t), label_count, fp_train_labels);
  print_image_labels(&images[0], labels[0]);

  free(images);
  free(labels);
  fclose(fp_train_images);
  fclose(fp_train_labels);
  return 0;
}
