#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <SDL/SDL.h>

/*********************
*  MILOUD BOUTOUBA   *
*                    *
**********************/

struct data_mnist{
  uint8_t *image;
  uint8_t label;
};
typedef struct data_mnist data_mnist;

struct cfg_mnist{
  uint32_t magic;
  uint32_t nb_data;
  uint32_t dim_x;
  uint32_t dim_y;
  data_mnist *data;
};
typedef struct cfg_mnist cfg_mnist;

struct node{
  uint8_t *w;
  double act;
  uint8_t label;
};
typedef struct node node;

struct net_cfg{
  int nb_node;
  int ligne;
  int colonne;
};
typedef struct net_cfg net_cfg;

struct bmu{
  int bmu_l;
  int bmu_c;
  struct bmu *suiv;
};
typedef struct bmu bmu;

struct bmu_Hdr{
  int cpt;
  struct bmu *first;
  struct bmu *last;
};
typedef struct bmu_Hdr bmu_Hdr;


void lecture(char*nom, cfg_mnist *config_mnist);
int trf(int nm);

uint8_t* image_moy(cfg_mnist config_mnist);
uint8_t gen_pixel(uint8_t pixel, uint8_t max, uint8_t min);
net_cfg configure_network(int nb_img);
node **init_network(uint8_t* img_moyenne, uint8_t max, uint8_t min, net_cfg N, cfg_mnist config_mnist);

void pause();
int rayon_vng(int nb_node);
void shuffle(int *indice, int nb_vec);
int* init_tab_indice(int nb_vec);
void neighborhood(node **Net, net_cfg N, int rayon, double alpha, bmu *winner, uint8_t* vec, int dim,int*ii,int*kk,int*jj,int*ll);
void apprentissage_SDL(node** Net, net_cfg N, cfg_mnist config_mnist);
double dist_euclid(uint8_t *img1, uint8_t *img2);
bmu *selectBMU(uint8_t* img, node** Net, net_cfg N);
