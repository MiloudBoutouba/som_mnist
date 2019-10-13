#MILOUD BOUTOUBA

CC = gcc -Wall
SRC = main.c initialise.c network.c apprentissage.c
OBJ = main.o initialise.o network.o apprentissage.o

SOM_MNIST:	$(OBJ) SOM_MNIST.h
	$(CC) -o SOM_MNIST $(OBJ) -lm -lSDL
%.o:	%.c SOM_MNIST.h
	$(CC) -c $<

clean:
	rm -rf *.o *~
