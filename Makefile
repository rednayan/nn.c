CC = gcc
CFLAGS = -Wall -Wextra -g
LDLIBS = -lm

TARGET = main
OBJS = main.o

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS)
