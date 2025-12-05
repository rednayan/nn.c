#ifndef NN_H
#define NN_H
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  uint8_t pixels[28 * 28];
} Image;

typedef struct {
  int rows;
  int cols;
  float *data;
} Matrix;

typedef struct {
  int rows;
  int cols;
  uint8_t *data;
} Matrix_uint8;

typedef struct {
  int n;
  int channels;
  int height;
  int width;
  float *data;
} Tensor;

Tensor *tensor_create(int n, int channels, int height, int width);
Tensor *tensor_conv2d(Tensor *t, Tensor *filter_t, int padding, int stride);
Tensor *tensor_randomize(Tensor *t);
void tensor_free(Tensor *t);

Matrix *matrix_create(int rows, int cols);
Matrix_uint8 *matrix_create_uint8(int rows, int cols);
Matrix *matrix_multiply(Matrix *x, Matrix *y);
Matrix *matrix_randomize(Matrix *mat);
Matrix *matrix_transpose(Matrix *mat);
Matrix *matrix_flatten(Image *img);
void matrix_free(Matrix *m);
void matrix_free_uint8(Matrix_uint8 *m);
void matrix_print(Matrix *mat);

Matrix *matrix_relu(Matrix *mat);
Matrix *matrix_softmax(Matrix *mat);
Matrix *matrix_one_hot(int label, int size);
int arg_max(Matrix *mat);

uint32_t fbytes(uint32_t val);
void matrix_save_bin(Matrix *mat, char *file_name);
Matrix *matrix_load_bin(char *file_name);
void matrix_save_to_header(Matrix *mat, char *array_name, FILE *file);
void matrix_save_to_header_uint8(Matrix_uint8 *mat, char *array_name,
                                 FILE *file);
Matrix *matrix_create_from_header(const float *data, int rows, int cols);

void print_image_labels(Image *image, uint8_t label);
void print_image_matrix(Matrix *mat);
#endif
