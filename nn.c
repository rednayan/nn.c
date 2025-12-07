#include "nn.h"
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

Tensor *tensor_create(int n, int channels, int height, int width)
{
	Tensor *t = (Tensor *)malloc(sizeof(Tensor));
	t->n = n;
	t->channels = channels;
	t->height = height;
	t->width = width;
	t->data = (float *)calloc(n * height * width, sizeof(float));
	return t;
}

Tensor *tensor_randomize(Tensor *t)
{
	for (int i = 0; i < t->n * t->channels * t->height * t->channels; ++i) {
		t->data[i] = ((float)rand() / RAND_MAX) - 0.5f;
	}
	return t;
}

// 1D index from 4D corordinates Formula: n*(C*H*W) + c*(H*W) + h*W + w
long long get_flat_index(int n, int c, int h, int w, int channels, int height,
			 int width)
{
	return (long long)n * (channels * height * width) +
	       (long long)c * (height * width) + (long long)h * (width) + w;
}

Tensor *tensor_conv2d(Tensor *input_t, Tensor *filter_t, int padding,
		      int stride)
{
	if (input_t->channels != filter_t->channels) {
		printf("Error: Input depth %d != Filter depth %d",
		       input_t->channels, filter_t->channels);
	}

	int size_input_t =
	    input_t->n * input_t->channels * input_t->height * input_t->width;
	int size_filter_t = filter_t->n * filter_t->channels *
			    filter_t->height * filter_t->width;

	// output_size = ((input - kernel) + 2 * padding) / stride
	int output_height =
	    ((input_t->height - filter_t->height) + 2 * padding) / stride + 1;
	int output_width =
	    ((input_t->width - filter_t->width) + 2 * padding) / stride + 1;

	printf("output_height: %d\n", output_height);
	printf("output_width: %d\n", output_width);

	Tensor *output_t =
	    tensor_create(input_t->n, filter_t->n, output_height, output_width);

	for (int b = 0; b < input_t->n; ++b) {
		for (int o = 0; o < filter_t->n; ++o) {
			for (int y = 0; y < output_t->height; ++y) {
				for (int x = 0; x < output_t->width; ++x) {
					float sum = 0.0f;
					for (int ic = 0; ic < input_t->channels;
					     ++ic) {
						for (int ky = 0;
						     ky < filter_t->height;
						     ++ky) {
							for (int kx = 0;
							     kx <
							     filter_t->width;
							     ++kx) {

								int in_y =
								    (y *
								     stride) -
								    padding +
								    ky;
								int in_x =
								    (x *
								     stride) -
								    padding +
								    kx;
								if (in_y >= 0 &&
								    in_y <
									input_t
									    ->height &&
								    in_x >= 0 &&
								    in_x <
									input_t
									    ->width) {
									long long input_idx = get_flat_index(
									    b,
									    ic,
									    in_y,
									    in_x,
									    input_t
										->channels,
									    input_t
										->height,
									    input_t
										->width);
									float val_in =
									    input_t
										->data
										    [input_idx];

									long long filter_idx = get_flat_index(
									    o,
									    ic,
									    ky,
									    kx,
									    filter_t
										->channels,
									    filter_t
										->height,
									    filter_t
										->width);

									float val_filter =
									    filter_t
										->data
										    [filter_idx];

									sum +=
									    val_in *
									    val_filter;
								}
							}
						}
					}
					long long out_idx = get_flat_index(
					    b, o, y, x, output_t->channels,
					    output_t->height, output_t->width);
					output_t->data[out_idx] = sum;
				}
			}
		}
	}
	return output_t;
}

Tensor *tensor_maxpool(Tensor *t, int stride)
{
	int out_h = t->height / stride;
	int out_w = t->width / stride;
	Tensor *output_t = tensor_create(t->n, t->channels, out_h, out_w);

	for (int b = 0; b < t->n; ++b) {
		for (int c = 0; c < t->channels; ++c) {
			for (int i = 0; i < out_h; ++i) {
				for (int j = 0; j < out_w; ++j) {
					float max_val = -INFINITY;
					for (int ki = 0; ki < stride; ++ki) {
						for (int kj = 0; kj < stride;
						     ++kj) {
							int row =
							    i * stride + kj;
							int col =
							    j * stride + kj;
							long long idx =
							    get_flat_index(
								b, c, row, col,
								t->channels,
								t->height,
								t->width);
							if (t->data[idx] >
							    max_val) {
								max_val =
								    t->data
									[idx];
							}
						}
					}
					long long out_idx = get_flat_index(
					    b, c, i, j, output_t->channels,
					    output_t->height, output_t->width);
					output_t->data[out_idx] = max_val;
				}
			}
		}
	}
	return output_t;
}

Tensor *tensor_relu(Tensor *t)
{
	int tensor_size = t->channels * t->n * t->height * t->width;
	for (int i = 0; i < tensor_size; ++i) {
		if (t->data[i] < 0.0f) {
			t->data[i] = 0.0f;
		}
	}
	return t;
}

Matrix *tensor_flatten_to_matrix(Tensor *t)
{
	int features = t->channels * t->height * t->width;
	Matrix *m = matrix_create(t->n, features);
	int total_elements = t->n * features;
	for (int i = 0; i < total_elements; ++i) {
		m->data[i] = t->data[i];
	}
	return m;
}

Matrix *matrix_create(int rows, int cols)
{
	Matrix *mat = (Matrix *)malloc(sizeof(Matrix));
	mat->rows = rows;
	mat->cols = cols;
	mat->data = (float *)calloc(rows * cols, sizeof(float));
	return mat;
}

Matrix_uint8 *matrix_create_uint8(int rows, int cols)
{
	Matrix_uint8 *mat = (Matrix_uint8 *)malloc(sizeof(Matrix_uint8));
	mat->rows = rows;
	mat->cols = cols;
	mat->data = (uint8_t *)calloc(rows * cols, sizeof(uint8_t));
	return mat;
}

Matrix *matrix_multiply(Matrix *x, Matrix *y)
{
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

Matrix *matrix_randomize(Matrix *mat)
{
	for (int i = 0; i < mat->rows; ++i) {
		for (int j = 0; j < mat->cols; ++j) {
			mat->data[i * mat->cols + j] =
			    ((float)rand() / RAND_MAX) - 0.5;
		}
	}
	return mat;
}

Matrix *matrix_transpose(Matrix *mat)
{
	Matrix *transposed_matrix = matrix_create(mat->cols, mat->rows);
	for (int i = 0; i < mat->rows; ++i) {
		for (int j = 0; j < mat->cols; ++j) {
			int src_index = i * mat->cols + j;
			int dst_index = j * transposed_matrix->cols + i;
			transposed_matrix->data[dst_index] =
			    mat->data[src_index];
		}
	}
	return transposed_matrix;
}

Matrix *matrix_flatten(Image *img)
{
	Matrix *mat = matrix_create(1, 784);

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			mat->data[i * 28 + j] = img->pixels[i * 28 + j] / 255.0;
		}
	}
	return mat;
}

void matrix_free(Matrix *m)
{
	if (m != NULL) {
		if (m->data != NULL) {
			free(m->data);
		}
		free(m);
	}
}

void matrix_free_uint8(Matrix_uint8 *m)
{
	if (m != NULL) {
		if (m->data != NULL) {
			free(m->data);
		}
		free(m);
	}
}

void tensor_free(Tensor *t)
{
	if (t != NULL) {
		if (t->data != NULL) {
			free(t->data);
		}
		free(t);
	}
}

void matrix_print(Matrix *mat)
{
	int rows = mat->rows;
	int cols = mat->cols;

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			printf(" %f", mat->data[i * cols + j]);
		}
		printf("\n");
	}
}

Matrix *matrix_relu(Matrix *mat)
{
	for (int i = 0; i < mat->rows; ++i) {
		for (int j = 0; j < mat->cols; ++j) {
			if (mat->data[i * mat->cols + j] < 0.0f) {
				mat->data[i * mat->cols + j] = 0.0f;
			}
		}
	}
	return mat;
}

Matrix *matrix_softmax(Matrix *mat)
{
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

Matrix *matrix_one_hot(int label, int size)
{
	Matrix *mat = matrix_create(1, size);
	mat->data[label] = 1.0f;
	return mat;
}

int arg_max(Matrix *mat)
{
	int max_index = 0;
	int mat_size = mat->rows * mat->cols;
	for (int i = 0; i < mat_size; ++i) {
		if (mat->data[i] > mat->data[max_index]) {
			max_index = i;
		}
	}
	return max_index;
}

uint32_t fbytes(uint32_t val)
{
	return ((val >> 24) & 0x000000FF) | ((val >> 8) & 0x0000FF00) |
	       ((val << 8) & 0x00FF0000) | ((val << 24) & 0xFF000000);
}

void matrix_save_bin(Matrix *mat, char *file_name)
{
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

Matrix *matrix_load_bin(char *file_name)
{
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

void matrix_save_to_header(Matrix *mat, char *array_name, FILE *file)
{
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
				 FILE *file)
{
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

Matrix *matrix_create_from_header(const float *data, int rows, int cols)
{
	Matrix *m = matrix_create(rows, cols);
	for (int i = 0; i < rows * cols; i++) {
		m->data[i] = data[i];
	}
	return m;
}

void print_image_labels(Image *image, uint8_t label)
{
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

void print_image_matrix(Matrix *mat)
{
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
