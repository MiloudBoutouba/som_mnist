#include "SOM_MNIST.h"

/*********************
*  MILOUD BOUTOUBA   *
*                    *
**********************/

int trf(int nm){
  uint32_t c1,c2,c3,c4;

  c1 = nm >> 24;
  c2 = (nm<<8)>>16;
  c3 = ((nm>>8)<<24)>>8;
  c4 = nm << 24;

  return ((c1+c2+c3+c4)<<16)>>16; //The first 2 bytes are always 0 in MNIST
}

void lecture(char*nom, cfg_mnist *config_mnist){
  uint8_t byte;
  uint32_t nm;
  int i,j;

  FILE* fichier = NULL;
  fichier = fopen(nom,"r");
  assert(fichier);


  fread(&nm, sizeof(nm), 1, fichier); config_mnist->magic = trf(nm);
  fread(&nm, sizeof(nm), 1, fichier); config_mnist->nb_data = trf(nm);
  fread(&nm, sizeof(nm), 1, fichier); config_mnist->dim_x = trf(nm);
  fread(&nm, sizeof(nm), 1, fichier); config_mnist->dim_y = trf(nm);

  config_mnist->data = malloc(sizeof(config_mnist->data[0])*config_mnist->nb_data);

  for(j=0; j<config_mnist->nb_data; j++){
    config_mnist->data[j].image = malloc(sizeof(config_mnist->data->image[0])*config_mnist->dim_x* config_mnist->dim_y);
  
    for(i=0; i<config_mnist->dim_x*config_mnist->dim_y; i++){
      fread(&byte, sizeof(byte), 1, fichier);
      config_mnist->data[j].image[i] = byte;
    }
  }
}