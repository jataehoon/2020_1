#include<stdio.h>
#include<stdlib.h>

FILE* fin;
FILE* fout;

typedef struct Array Array;

struct Array {
    int size;
    int* values;
};

Array* CreateArray(int size);
void QuickSort(int array[], int size, int PIVOT, int SIZE);
void Partition(int PIVOT, int pivot, int size, int SIZE, int array[]);
void PrintArray(int array[], int size);
void DeleteArray(Array* array);

int main(int argc, char* argv[]){
    if (argc == 1) {
        fin = fopen("input.txt", "r");
        fout = fopen("output.txt", "w");
    }
    else {
        fin = fopen(argv[1], "r");
        fout = fopen(argv[2], "w");
    }
    int size, i;
    fscanf(fin, "%d", &size);
    int* array = (int*)malloc(sizeof(int) * size);
    for (i = 0; i < size; i++) {
        fscanf(fin, "%d", &array[i]);
    }
    QuickSort(array, size, 0, size);
    PrintArray(array, size);
    free(array);
}

void QuickSort(int array[], int size, int PIVOT, int SIZE){
    int pivot, temp;
    if (size>1) {
        pivot = array[size-1];
        int left = -1;
        int right = size-1;
        int LF, RGT;
        for (LF=0; LF<size; LF++) if (pivot<array[LF]) break;
        for (RGT=size-1; RGT>=0; RGT--) if (pivot==array[RGT]) break;
        if (PIVOT>0 && LF<RGT) PIVOT = -1;
        if (PIVOT==0) fprintf(fout,"Left: %d, Right: %d, Pivot: %d\n",0,RGT,pivot);
        else if (PIVOT>0 && pivot+size<SIZE){
            fprintf(fout,"Left: %d, Right: %d, Pivot: %d\n",pivot-1,pivot+size-2,pivot);
        }
        else if (PIVOT>0 && pivot+size>SIZE){
            fprintf(fout,"Left: %d, Right: %d, Pivot: %d\n",pivot-size+1,pivot,pivot);
        }
        else if (PIVOT<0 && pivot+size<SIZE){
            fprintf(fout,"Left: %d, Right: %d, Pivot: %d\n",pivot,pivot+size-1,pivot);
        }
        Partition(PIVOT, pivot, size, SIZE, array);
        while (1) {
            while (array[++left]<pivot);
            while (array[--right]>pivot);
            if (left>=right) break;
            temp = array[left];
            array[left] = array[right];
            array[right] = temp;
            Partition(PIVOT, pivot, size, SIZE, array);
        }
        temp = array[left];
        array[left] = array[size-1];
        array[size-1] = temp;
        if (array[left]!=array[size-1]){
            Partition(PIVOT, pivot, size, SIZE, array);
        }
        QuickSort(array, left, PIVOT, SIZE);
        if (left>right) PIVOT++;
        QuickSort(array+left+1, size-left-1, PIVOT, SIZE);
    }
}

void Partition(int PIVOT, int pivot, int size, int SIZE, int array[]){
    if (PIVOT==0) {
        for (int i=0; i<size; i++) fprintf(fout, "%-3d", array[i]);
        for (int i=size; i<SIZE; i++) fprintf(fout, "-  ");
        fprintf(fout, "\n");
    }
    else if (PIVOT>0 && pivot+size<SIZE) {
        for (int i=0; i<pivot-1; i++) fprintf(fout, "-  ");
        for (int i=0; i<size; i++) fprintf(fout, "%-3d", array[i]);
        for (int i=size; i<SIZE-pivot+1; i++) fprintf(fout, "-  ");
        fprintf(fout, "\n");
    }
    else if (PIVOT>0 && pivot+size>SIZE) {
        for (int i=0; i<pivot-size+1; i++) fprintf(fout, "-  ");
        for (int i=0; i<size; i++) fprintf(fout, "%-3d", array[i]);
        fprintf(fout, "\n");
    }
    else if (PIVOT<0 && pivot+size<SIZE) {
        for (int i=0; i<pivot; i++) fprintf(fout, "-  ");
        for (int i=0; i<size; i++) fprintf(fout, "%-3d", array[i]);
        for (int i=SIZE-size+1; i<SIZE; i++) fprintf(fout, "-  ");
        fprintf(fout, "\n");
    }
}

void PrintArray(int array[], int size) {
    fprintf(fout, "\nSorting Result: \n");
    for (int i=0; i<size; i++) {
        fprintf(fout, "%-3d", array[i]);
    }
    fprintf(fout, "\n");
}

Array* CreateArray(int size) {
    int* Arr = malloc(sizeof(int)*size);
}

void DeleteArray(Array* array){
    free(array);
}
