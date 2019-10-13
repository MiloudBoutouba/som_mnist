#include "SOM_MNIST.h"

/*********************
*  MILOUD BOUTOUBA   *
*                    *
**********************/

void pause()
{
    int continuer = 1;
    SDL_Event event;
 
    while (continuer)
    {
        SDL_WaitEvent(&event);
        switch(event.type)
        {
            case SDL_QUIT:
                continuer = 0;
        }
    }
}


int rayon_vng(int nb_node){
  int rayon = 0;
  int nb_vng = 0; 

  while(nb_vng < nb_node *0.50){
    rayon++;
    nb_vng += 8 * rayon;
  }
  return rayon;
}

void shuffle(int *indice, int nb_vec){
  int randindice = 0;
  int i,tmp;

  for(i=0; i<nb_vec; i++){
    randindice = rand()%nb_vec;

    tmp = indice[i];
    indice[i] = indice[randindice];
    indice[randindice] = tmp;
  }
}

int* init_tab_indice(int nb_vec){
  int i; 
  int *indice = malloc(sizeof(indice[0])*nb_vec);
  for(i=0; i<nb_vec; i++)
    indice[i] = i;
  return indice;
}



void neighborhood(node **Net, net_cfg N, int rayon, double alpha, bmu *winner, uint8_t* vec, int dim,
  int *x1,int *x2,int *y1,int *y2) //position dédut et fin du voisinage
{

  int i,j,k,l,m;
  i = winner->bmu_l-rayon; if(i < 0) i = 0;
  j = winner->bmu_c-rayon; if(j < 0) j = 0;
  k = winner->bmu_l+rayon; if(k >= N.ligne) k = N.ligne-1;
  l = winner->bmu_c+rayon; if(l >= N.colonne) l = N.colonne-1;
  *x1 = i; *x2 = k; *y1 = j; *y2 = l;

  for(i=i; i<=k; i++){
    j = winner->bmu_c-rayon; if(j < 0) j = 0;
    for(j=j; j<=l; j++)
      for(m=0; m<dim; m++)
        Net[i][j].w[m] += (alpha * (vec[m] - Net[i][j].w[m]))+ 0.5;
  }   
}

void affiche(uint8_t* im){
  int i;
  for(i=0; i<28*28; i++){
    if(!(i%28)) printf("\n");
    printf("%3d ", im[i]);
  }
}

void apprentissage_SDL(node** Net, net_cfg N, cfg_mnist config_mnist){
  SDL_Surface *ecran = NULL, *pixel = NULL;
  SDL_Rect position;
  int i,j,k,l,m, x1,y1,x2,y2, rayon;
  bmu *winner;
  double alpha_init, alpha;
  int *indice;

  int it_max = config_mnist.nb_data > 10000 ? config_mnist.nb_data * 3: config_mnist.nb_data * 20;

  int ord = it_max * 0.25;
  int aff = it_max - ord;
  int it, index;
  printf("Iteration : MAX %d - ORD %d - AFF %d \n", it_max, ord,aff);

  SDL_Init(SDL_INIT_VIDEO);

  ecran = SDL_SetVideoMode(28*N.colonne, 28*N.ligne, 32, SDL_HWSURFACE);

  pixel = SDL_CreateRGBSurface(SDL_HWSURFACE, 1, 1, 32, 0, 0, 0, 0);
  SDL_WM_SetCaption("MNIST - SOM", NULL);

  SDL_FillRect(ecran, NULL, SDL_MapRGB(ecran->format, 140, 140, 140));

  printf("Configuration Réseau de Neurones : %d Lignes %d Colonnes\n", N.ligne, N.colonne);

  for(l=0; l<N.ligne; l++){
    for(m=0; m<N.colonne; m++){
      k=0;
      for (i=l*28 ; i <28*(l+1); i++){
        for(j=m*28; j<28*(m+1); j++){
          position.x = j; 
          position.y = i;
          SDL_FillRect(pixel, NULL, SDL_MapRGB(ecran->format,Net[l][m].w[k], Net[l][m].w[k], Net[l][m].w[k]));
          SDL_BlitSurface(pixel, NULL, ecran, &position);
          k++;
        }
      }
    }    
  }
  SDL_Flip(ecran);

  x1 = 0; y1 = 0;x2 = 0; y2 = 0;

  rayon = rayon_vng(N.nb_node);
  indice = init_tab_indice(config_mnist.nb_data);

  alpha_init = (0.9 - 0.7) * ((double)rand()/RAND_MAX) + 0.7;

  for(it=0; it<ord; it++){

    if(it != 0)
      if(!(it%(ord/rayon))){
        rayon--;
        printf("Rayon --> %d Alpha --> %lf\n", rayon,alpha);
      }


    alpha = alpha_init * (1 - ((double)it/(double)ord));
    shuffle(indice, config_mnist.nb_data);
    index = rand()%config_mnist.nb_data;
    winner = selectBMU(config_mnist.data[indice[index]].image, Net,N);
    neighborhood(Net, N, rayon, alpha, winner, config_mnist.data[indice[index]].image, 28*28,&x1,&x2,&y1,&y2);

    for(l=x1; l<=x2; l++){
      for(m=y1; m<=y2; m++){    
        k=0;
        for (i=l*28 ; i <28*(l+1); i++){
          for(j=m*28; j<28*(m+1); j++){
            position.x = j; 
            position.y = i;
            SDL_FillRect(pixel, NULL, SDL_MapRGB(ecran->format, Net[l][m].w[k], Net[l][m].w[k], Net[l][m].w[k]));
            SDL_BlitSurface(pixel, NULL, ecran, &position);
            k++;
          }
        }
      }
    }
    SDL_Flip(ecran);
  }

  printf("Fin Phase ordonnancement ! %d itérations\nAppuyer sur ENTREE \n",it);
  getchar();

  rayon = 1;
  alpha_init /= 10;
  printf("Debut Phase d'affinage : \nRayon : %d Alpha : %lf\n",rayon,alpha_init);
  
  for(it=0; it<aff; it++){
    alpha = alpha_init * (1 - ((double)it/(double)aff));

    shuffle(indice, config_mnist.nb_data);
    index = rand()%config_mnist.nb_data;
    winner = selectBMU(config_mnist.data[indice[index]].image, Net,N);
    neighborhood(Net, N, rayon, alpha_init, winner, config_mnist.data[indice[index]].image, 28*28,&x1,&x2,&y1,&y2);
    for(l=x1; l<=x2; l++){
      for(m=y1; m<=y2; m++){      
        k=0;
        for (i=l*28 ; i <28*(l+1); i++){
          for(j=m*28; j<28*(m+1); j++){
            position.x = j; 
            position.y = i;
            SDL_FillRect(pixel, NULL, SDL_MapRGB(ecran->format, Net[l][m].w[k], Net[l][m].w[k], Net[l][m].w[k]));
            SDL_BlitSurface(pixel, NULL, ecran, &position);
            k++;
          }
        }
      }
    }
      SDL_Flip(ecran);
  }
  printf("Fin Phase affinage ! %d it\n",it);

  printf("~ FIN ~\n");

  pause();

  SDL_FreeSurface(pixel);
  SDL_Quit();
}



double dist_euclid(uint8_t *img1, uint8_t *img2){
  int distance = 0;
  int i;
  for(i=0; i<28*28; i++)
    distance += (int)(img1[i]-img2[i])*(int)(img1[i]-img2[i]);

  return sqrt(distance);
}





bmu *selectBMU(uint8_t* img, node** Net, net_cfg N){
  double min = 10000.0, distance = 0;
  bmu *winner;
  int i,j, rand_bmu;
  bmu_Hdr header;
  header.cpt = 0; header.first = NULL; header.last = NULL;

  for(i=0; i<N.ligne; i++){
    for(j=0; j<N.colonne; j++){
      distance = dist_euclid(img, Net[i][j].w);
      if(distance <= min){
        if(distance < min){
          free(header.first);
          header.first = NULL; header.last = NULL; header.cpt = 0;
        }
        bmu *best = malloc(sizeof(best[0]));
        best->bmu_l = i; best->bmu_c = j; best->suiv = NULL;

        if(header.first == NULL){
          header.first = best; header.last = best;
        }else{
          header.last->suiv = best; header.last = best;
        }
        header.cpt++;
        min = distance;
      }
    }
  }
  winner = header.first;
  
  if(header.cpt == 1)
    return winner;

  else if(header.cpt > 1){
    rand_bmu = rand()%header.cpt;
    while(rand_bmu){
      winner = winner->suiv;
      rand_bmu--;
    }
    return winner;
  }
  return NULL;
}
