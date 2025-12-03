#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

uint32_t swap_endian(uint32_t val) {
  return ((val >> 24) & 0x000000FF) | ((val >> 8) & 0x0000FF00) |
         ((val << 8) & 0x00FF0000) | ((val << 24) & 0xFF000000);
}

void save_pgm(uint8_t *buffer, int rows, int cols, const char *filename) {
  FILE *fp = fopen(filename, "wb");
  // P5 header: Binary Grey, Width, Height, Max Value
  fprintf(fp, "P5\n%d %d\n255\n", cols, rows);
  fwrite(buffer, sizeof(uint8_t), rows * cols, fp);
  fclose(fp);
  printf("Saved %s\n", filename);
}

int main() {
  char *input_filepath = "/home/syien/Documents/dev/nn/MNIST_ORG/";
  char input_images[256];

  sprintf(input_images, "%strain-images.idx3-ubyte", input_filepath);
  FILE *fp = fopen(input_images, "rb");
  if (fp == NULL) {
    perror("Error opening images file");
    return 1;
  }

  uint32_t magic, num_images, rows, cols, magic_label, num_items, data;

  fread(&magic, sizeof(uint32_t), 1, fp);
  fread(&num_images, sizeof(uint32_t), 1, fp);
  fread(&rows, sizeof(uint32_t), 1, fp);
  fread(&cols, sizeof(uint32_t), 1, fp);

  magic = swap_endian(magic);
  num_images = swap_endian(num_images);
  rows = swap_endian(rows);
  cols = swap_endian(cols);

  if (magic != 2051) {
    printf("Invalid magic number!");
    fclose(fp);
    return 1;
  }
  char input_labels[256];

  sprintf(input_labels, "%strain-labels.idx1-ubyte", input_filepath);
  FILE *fp_label = fopen(input_labels, "rb");
  if (fp_label == NULL) {
    perror("Error opening lable file");
    return 1;
  }

  fread(&magic_label, sizeof(uint32_t), 1, fp_label);
  fread(&num_items, sizeof(uint32_t), 1, fp_label);

  magic_label = swap_endian(magic_label);
  num_items = swap_endian(num_items);

  if (magic_label != 2049) {
    printf("Invalid magic number for label file");
    fclose(fp_label);
    return 1;
  }

  int image_size = rows * cols;
  uint8_t *image_buffer = (uint8_t *)malloc(image_size * sizeof(uint8_t));
  uint8_t label_data;
  int images_to_save = 10;

  char file_name[256];
  char *file_path = "/home/syien/Documents/dev/nn/output/";

  for (int i = 0; i < images_to_save; i++) {
    fread(image_buffer, sizeof(uint8_t), image_size, fp);
    fread(&label_data, 1, 1, fp_label);
    sprintf(file_name, "%s/train_%d_label_%d.pgm", file_path, i, label_data);
    save_pgm(image_buffer, rows, cols, file_name);
  }

  free(image_buffer);
  fclose(fp);

  return 0;
}
