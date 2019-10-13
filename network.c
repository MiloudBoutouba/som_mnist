#include "SOM_MNIST.h"

/*********************
*  MILOUD BOUTOUBA   *
*                    *
**********************/

uint8_t* image_moy(cfg_mnist config_mnist){
  uint8_t *img_moyenne = malloc(sizeof(img_moyenne[0])*config_mnist.dim_x*config_mnist.dim_y);
  int i,j;
  uint32_t tmp;

  for(i=0; i<config_mnist.dim_x*config_mnist.dim_y; i++){
    tmp = 0;
    for(j=0; j<config_mnist.nb_data; j++)
      tmp += config_mnist.data[j].image[i];

    img_moyenne[i] = (uint32_t)(tmp/config_mnist.nb_data);
  }
  return img_moyenne;
}

uint8_t gen_pixel(uint8_t pixel, uint8_t max, uint8_t min){
  int pixel_back;

  if(pixel == 0) return 0;

  pixel_back = (int)(((pixel+max) - (pixel-min)) * ((double)rand()/RAND_MAX) + (pixel-min));

  if(pixel_back < 0) pixel_back = 0;
  if(pixel_back > 255) pixel_back = 255;
  
  return (uint8_t)pixel_back;
}
net_cfg configure_network(int nb_img){
  double or = 1.6; /*Nombre d'OR 1.61803398875 */
  int nb_node_max = 2*sqrt(nb_img);
  float colonne = 0,colonne_max = 0, ligne = 0, ligne_max = 0;
  net_cfg N;

  while(colonne_max * ligne_max <= nb_node_max){
    colonne = colonne_max;
    ligne = ligne_max;

    ligne_max++;
    colonne_max = ligne_max * or;
  }

  colonne +=0.5;
  ligne +=0.5;

  N.colonne = (int)colonne;
  N.ligne = (int)ligne - 2;
  N.nb_node = N.ligne * N.colonne;

  return N;
}

node **init_network(uint8_t* img_moyenne, uint8_t max, uint8_t min, net_cfg N, cfg_mnist config_mnist){
  int i,j,k;
  node **Net = malloc(sizeof(Net[0])*N.ligne);
  for(i=0; i<N.ligne; i++)
    Net[i] = malloc(sizeof(Net[i][0])*N.colonne);

  for(i=0; i<N.ligne; i++)    
    for(j=0; j<N.colonne; j++){
      Net[i][j].w = malloc(sizeof(Net[i][j].w[0])*config_mnist.dim_x*config_mnist.dim_y);
      for(k=0;k<config_mnist.dim_x*config_mnist.dim_y; k++)
        Net[i][j].w[k] = gen_pixel(img_moyenne[k], max, min);
    }

  return Net;
}