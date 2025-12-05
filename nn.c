#include "nn.h"
#include <stdint.h>
#include <string.h>

Matrix *matrix_create(int rows, int cols) {
  Matrix *mat = (Matrix *)malloc(sizeof(Matrix));
  mat->rows = rows;
  mat->cols = cols;
  mat->data = (float *)calloc(rows * cols, sizeof(float));
  return mat;
}

Matrix_uint8 *matrix_create_uint8(int rows, int cols) {
  Matrix_uint8 *mat = (Matrix_uint8 *)malloc(sizeof(Matrix_uint8));
  mat->rows = rows;
  mat->cols = cols;
  mat->data = (uint8_t *)calloc(rows * cols, sizeof(uint8_t));
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

void matrix_free(Matrix *m) {
  if (m != NULL) {
    if (m->data != NULL) {
      free(m->data);
    }
    free(m);
  }
}

void matrix_free_uint8(Matrix_uint8 *m) {
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

uint32_t fbytes(uint32_t val) {
  return ((val >> 24) & 0x000000FF) | ((val >> 8) & 0x0000FF00) |
         ((val << 8) & 0x00FF0000) | ((val << 24) & 0xFF000000);
}

void matrix_save_bin(Matrix *mat, char *file_name) {
  FILE *file = fopen(file_name, "wb");
  if (file == NULL) {
    perror("Error: file not found.");
    return;
  }

  fwrite(&mat->rows, sizeof(int), 1, file);
  fwrite(&mat->cols, sizeof(int), 1, file);

  fwrite(mat->data, sizeof(float), mat->rows * mat->cols, file);
  printf("Success: Saved matrix at: %s\n", file_name);
  fclose(file);
}

Matrix *matrix_load_bin(char *file_name) {
  FILE *file = fopen(file_name, "rb");
  if (file == NULL) {
    perror("Error: File not found.");
    return NULL;
  }
  int rows, cols;
  fread(&rows, sizeof(int), 1, file);
  fread(&cols, sizeof(int), 1, file);

  Matrix *mat = matrix_create(rows, cols);
  fread(mat->data, sizeof(float), rows * cols, file);
  printf("Success: Read matrix from %s\n", file_name);
  fclose(file);
  return mat;
}

void matrix_save_to_header(Matrix *mat, char *array_name, FILE *file) {
  fprintf(file, "#define ROWS_%s %d\n#define COLS_%s %d\n\n", array_name,
          mat->rows, array_name, mat->cols);

  fprintf(file, "const float %s[] = {\n ", array_name);

  for (int i = 0; i < mat->rows * mat->cols; ++i) {
    fprintf(file, "%.7ff", mat->data[i]);
    if (i < (mat->rows * mat->cols) - 1) {
      fprintf(file, ",");
    }
    if ((i + 1) % 10 == 0) {
      fprintf(file, "\n");
    }
  }
  fprintf(file, "}; \n\n");
}

void matrix_save_to_header_uint8(Matrix_uint8 *mat, char *array_name,
                                 FILE *file) {
  fprintf(file, "#define ROWS_%s %d\n#define COLS_%s %d\n\n", array_name,
          mat->rows, array_name, mat->cols);

  fprintf(file, "const uint8_t %s[] = {\n ", array_name);

  for (int i = 0; i < mat->rows * mat->cols; ++i) {
    fprintf(file, "%d", mat->data[i]);
    if (i < (mat->rows * mat->cols) - 1) {
      fprintf(file, ",");
    }
    if ((i + 1) % 10 == 0) {
      fprintf(file, "\n");
    }
  }
  fprintf(file, "}; \n\n");
}

Matrix *matrix_create_from_header(const float *data, int rows, int cols) {
  Matrix *m = matrix_create(rows, cols);
  for (int i = 0; i < rows * cols; i++) {
    m->data[i] = data[i];
  }
  return m;
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
