#include "SOM_MNIST.h"

/*********************
*  MILOUD BOUTOUBA   *
*                    *
**********************/

int main(int argc, char** argv){

  if(argc < 2){
    fprintf(stderr,"USAGE : %s fichier_images_ubyte\ni.e : %s train-images.idx3-ubyte\n",argv[0],argv[0]);
    return 0;
  }

  srand(time(NULL));
  cfg_mnist config_mnist;
  uint8_t *img_moyenne;
  net_cfg N;
  node** Net;

  lecture(argv[1],&config_mnist);
  img_moyenne = image_moy(config_mnist);

  N = configure_network(config_mnist.nb_data);
  Net = init_network(img_moyenne,10,5,N,config_mnist);
  apprentissage_SDL(Net, N, config_mnist);


  return 0;
}
