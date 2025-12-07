#include "nn.h"
#include "weights.h"
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <time.h>

#define EPOCH 1

void float_to_uint8_matrix(Matrix *float_weights, Matrix_uint8 *uint_weights)
{
	// TODO
}

void validate_model(int num_validation_images, Image *validation_images,
		    uint8_t *validation_labels)
{

	// Matrix *W1_bin = matrix_load_bin("W1.bin");
	// Matrix *W2_bin = matrix_load_bin("W2.bin");

	Matrix *W1_head = matrix_create(ROWS_W1_HEADER, COLS_W1_HEADER);
	Matrix *W2_head = matrix_create(ROWS_W2_HEADER, COLS_W2_HEADER);

	for (int j = 0; j < ROWS_W1_HEADER * COLS_W1_HEADER; ++j) {
		W1_head->data[j] = W1_HEADER[j];
	}

	for (int m = 0; m < ROWS_W2_HEADER * COLS_W2_HEADER; ++m) {
		W2_head->data[m] = W2_HEADER[m];
	}

	for (int i = 0; i < num_validation_images; ++i) {
		Matrix *input_validation =
		    matrix_flatten(&validation_images[i]);
		Matrix *input_val = matrix_multiply(input_validation, W1_head);
		Matrix *input_val_relu = matrix_relu(input_val);
		Matrix *input_val_2 = matrix_multiply(input_val_relu, W2_head);
		Matrix *pred_out_val = matrix_softmax(input_val_2);

		int prediction_output = arg_max(pred_out_val);
		printf("prediction: %d\n", prediction_output);
		print_image_labels(&validation_images[i], validation_labels[i]);

		matrix_free(input_validation);
		matrix_free(input_val_relu);
		matrix_free(pred_out_val);
	}
}
void train(Matrix *W1, Matrix *W2, int num_train_images, Image *train_images,
	   uint8_t *train_labels, int epoch)
{

	for (int i = 0; i < num_train_images; ++i) {
		Matrix *X = matrix_flatten(&train_images[i]);
		Matrix *X_raw = matrix_multiply(X, W1);
		Matrix *X1 = matrix_relu(X_raw);
		Matrix *X1W2 = matrix_multiply(X1, W2);
		Matrix *output = matrix_softmax(X1W2);

		Matrix *error_matrix = matrix_create(1, 10);
		Matrix *target_matrix = matrix_one_hot(train_labels[i], 10);
		for (int j = 0; j < 10; j++) {
			error_matrix->data[j] =
			    output->data[j] - target_matrix->data[j];
		}
		Matrix *transpose_X1 = matrix_transpose(X1);
		Matrix *grad2 = matrix_multiply(transpose_X1, error_matrix);
		Matrix *transpose_W2 = matrix_transpose(W2);
		Matrix *error_matrix_2 =
		    matrix_multiply(error_matrix, transpose_W2);

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
}

void test(Matrix *W1, Matrix *W2, int num_test_images, Image *test_images,
	  uint8_t *test_labels, int epoch)
{
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

int main()
{

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
	uint8_t *test_labels =
	    (uint8_t *)malloc(sizeof(uint8_t) * test_label_count);

	Image *validation_images =
	    (Image *)malloc(sizeof(Image) * num_validation_images);
	uint8_t *validation_labels =
	    (uint8_t *)malloc(sizeof(uint8_t) * test_label_count);

	fread(train_images, sizeof(Image), num_train_images, fp_train_images);
	fread(train_labels, sizeof(uint8_t), train_label_count,
	      fp_train_labels);

	fread(test_images, sizeof(Image), num_test_images, fp_test_images);
	fread(test_labels, sizeof(uint8_t), test_label_count, fp_test_labels);

	fread(validation_images, sizeof(Image), num_validation_images,
	      fp_test_images);
	fread(validation_labels, sizeof(uint8_t), validation_label_count,
	      fp_test_labels);

	// input(28X28) -> conv2d(26X26) -> maxpool(13X13) -> filters(8)
	// = 8 X 13 X 13
	Matrix *W1 = matrix_create(1352, 128);
	Matrix *W2 = matrix_create(128, 10);

	Tensor *filter_t = tensor_create(8, 1, 3, 3);
	tensor_randomize(filter_t);

	for (int i = 0; i < EPOCH; ++i) {
		for (int i = 0; i < 10; ++i) {
			Image *img = &train_images[i];
			Tensor *input_t = tensor_flatten(img);
			Tensor *input_t_conv2d =
			    tensor_conv2d(input_t, filter_t, 0, 1);
			tensor_relu(input_t_conv2d);
			Tensor *input_t_conv2d_maxpool =
			    tensor_maxpool(input_t_conv2d, 2);
			Matrix *flattened =
			    tensor_flatten_to_matrix(input_t_conv2d_maxpool);

			Matrix *hidden_raw = matrix_multiply(flattened, W1);
			matrix_relu(hidden_raw);

			Matrix *output = matrix_multiply(hidden_raw, W2);
			matrix_softmax(output);

			matrix_print(output);

			tensor_free(input_t);
			tensor_free(input_t_conv2d);
			tensor_free(input_t_conv2d_maxpool);
			matrix_free(flattened);
			matrix_free(hidden_raw);
			matrix_free(output);
		}
	}
	/*
	Matrix *W1 = matrix_randomize(matrix_create(784, 128));
	Matrix *W2 = matrix_randomize(matrix_create(128, 10));
	for (int i = 0; i < EPOCH; ++i) {
		train(W1, W2, num_train_images, train_images, train_labels, i);
		test(W1, W2, num_test_images, test_images, test_labels, i);
	}
	Matrix *W1 = matrix_randomize(matrix_create(784, 128));
	Matrix *W2 = matrix_randomize(matrix_create(128, 10));
	for (int i = 0; i < EPOCH; ++i) {
	  train(W1, W2, num_train_images, train_images, train_labels, i);
	  test(W1, W2, num_test_images, test_images, test_labels, i);
	}
	//  validate_model(num_validation_images, validation_images,
	//  validation_labels);

	matrix_save_bin(W1, "W1.bin");
	matrix_save_bin(W2, "W2.bin");
	printf("Success: Saved float weights bin files\n");

	char *header_file_name = "weights.h";
	FILE *header_file = fopen(header_file_name, "w");
	matrix_save_to_header(W1, "W1_HEADER", header_file);
	matrix_save_to_header(W2, "W2_HEADER", header_file);
	fclose(header_file);
	printf("Success: Saved float weights at header file: %s\n",
	header_file_name);

	char *header_uint8_file_name = "weights_uint8.h";
	FILE *header_uint8_file = fopen(header_uint8_file_name, "w");
	Matrix_uint8 *W1_uint8 = matrix_create_uint8(W1->rows, W1->cols);
	Matrix_uint8 *W2_uint8 = matrix_create_uint8(W2->rows, W2->cols);
	float_to_uint8_matrix(W1, W1_uint8);
	float_to_uint8_matrix(W2, W2_uint8);
	fprintf(header_uint8_file, "#include <stdint.h>\n");
	matrix_save_to_header_uint8(W1_uint8, "W1_HEADER", header_uint8_file);
	matrix_save_to_header_uint8(W2_uint8, "W2_HEADER", header_uint8_file);
	fclose(header_uint8_file);
	printf("Success: Saved uint8 weights at header file: %s\n",
	       header_uint8_file_name);
	*/

	// matrix_free(W1);
	// matrix_free(W2);
	// matrix_free_uint8(W1_uint8);
	// matrix_free_uint8(W2_uint8);

	free(train_images);
	free(train_labels);
	free(test_images);
	free(test_labels);
	fclose(fp_train_images);
	fclose(fp_train_labels);
	return 0;
}
