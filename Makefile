CC = gcc
CFLAGS = -Wall -Wextra -g
LDLIBS = -lm

TARGET = main
OBJS = main.o nn.o

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

main.o : nn.h
nn.o : nn.h

.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS)
