#include<stdio.h>
#include<stdlib.h>
#include<time.h>

FILE *fin;
FILE *fout;

typedef struct _DisjointSet {
    int leftright;
    int updown;
}DisjointSets;

void init(int i);
void Union(int i);
int find(int i);
void createMaze(int i);
void printMaze(int i);
void freeMaze(int i);
int *dim1;
char **dim2;
DisjointSets *x, *y;

int main(int argc, char* argv[]){
    srand((unsigned int)time(NULL));
    int i;
    if (argc == 1) {
        fin = fopen("input.txt", "r");
        fout = fopen("output.txt", "w");
    }
    else {
        fin = fopen(argv[1], "r");
        fout = fopen(argv[2], "w");
    }
    fscanf(fin, "%d", &i);
    init(i);
    Union(i);
    createMaze(i);
    printMaze(i);
    freeMaze(i);
    fclose(fin);
    fclose(fout);
    return 0;
}

void init(int k){
    dim1 = (int*)malloc(sizeof(int)*k*k);
    for (int i=1; i<=k*k; i++) dim1[i]=i;
}

void Union(int k){
    dim2 = (char**)malloc(sizeof(char*)*k*2);
    for (int i=0; i<=k*2; i++) dim2[i]=(char*)malloc(sizeof(char)*k*2);
    for (int i=0; i<=k*2; i++) {
        for (int j=0; j<=k*2; j++) {
            if (i%2 == 0){
                if (j%2) dim2[i][j]='-';
                else dim2[i][j]='+';
            }
            else {
                if (j%2) dim2[i][j]='^';
                else dim2[i][j]='|';
            }
        }
    }
    dim2[1][0]=' ';
    dim2[k*2-1][k*2]=' ';
    x = (DisjointSets*)malloc(sizeof(DisjointSets)*k*(k-1));
    y = (DisjointSets*)malloc(sizeof(DisjointSets)*k*(k-1));
    int cp1 = 0;
    for (int i=0; i<k; i++){
        for (int j=i*k+1; j<k*(i+1); j++){
            x[cp1].leftright = j;
            x[cp1++].updown = j+1;
        }
    }
    cp1 = 0;
    for (int i=1; i<k+1; i++){
        for (int j=i; j<=k*(k-1); j+=k){
            y[cp1].leftright = j;
            y[cp1++].updown = j+k;
        }
    }
}

int find(int i){
    if (i==dim1[i]) return i;
    else return dim1[i]=find(dim1[i]);
}

void createMaze(int k){
    int cp2;
    while (find(k*k)!=1){
        int ran = rand()%(k*(k-1)*2);
        if (ran >= k*(k-1)){
            ran-=k*(k-1);
            if (find(y[ran].leftright)==find(y[ran].updown)) cp2=0;
            else {
                if (dim1[y[ran].leftright]<dim1[y[ran].updown]) dim1[dim1[y[ran].updown]]=dim1[y[ran].leftright];
                else dim1[dim1[y[ran].leftright]]=dim1[y[ran].updown];
                cp2 = 1;
            }
            if (cp2) {
                int a = 2*(ran%(k-1)+1);
                int b = 1+ran/(k-1)*2;
                dim2[a][b]='^';
            }
        }
        else {
            if (find(x[ran].leftright)==find(x[ran].updown)) cp2=0;
            else {
                if (dim1[x[ran].leftright]<dim1[x[ran].updown]) dim1[dim1[x[ran].updown]]=dim1[x[ran].leftright];
                else dim1[dim1[x[ran].leftright]]=dim1[x[ran].updown];
                cp2 = 1;
            }
            if (cp2) {
                int a = 1+ran/(k-1)*2;
                int b = 2*(ran%(k-1)+1);
                dim2[a][b]=' ';
            }
        }
    }
}

void printMaze(int k){
    for (int i=0; i<k*2+1; i++){
        for (int j=0; j<k*2+1; j++){
            if (dim2[i][j]=='-') fprintf(fout, "---");
            else if (dim2[i][j]=='^') fprintf(fout, "   ");
            else fprintf(fout, "%c", dim2[i][j]);
        }
        fprintf(fout, "\n");
    }
}

void freeMaze(int k){
    free(x);
    free(y);
    for (int i=0; i<k*2+1; i++) free(dim2[i]);
    free(dim2);
    free(dim1);
}
