#include<stdio.h>
#include<stdlib.h>

FILE *fin;
FILE *fout;

typedef struct HeapStruct* PriorityQueue;
PriorityQueue heap;
struct HeapStruct { //구조체 생성
    int Capacity;
    int Size;
    int *Elements;
};
void CreateHeap(); //heap 생성
void Insert(PriorityQueue heap, int value); //삽입 함수
void Find(PriorityQueue heap, int value); //찾기 함수
void DeleteMax(PriorityQueue heap); //최대 원소 삭제 함수
void PrintHeap(PriorityQueue heap); //출력 함수

int main(int argc, char* argv[]){
    if (argc == 1){
        fin = fopen("input.txt", "r");
        fout = fopen("output.txt", "w");
    }
    else {
        fin = fopen(argv[1], "r");
        fout = fopen(argv[2], "w");
    }
    char cv;
    int key;
    CreateHeap(); //heap 생성
    while (!feof(fin)){
        cv = fgetc(fin);
        switch (cv) {
            case 'i': //i 받으면 삽입
                fscanf(fin, "%d", &key);
                Insert(heap, key);
                break;
            case 'd': //d 받으면 삭제
                DeleteMax(heap);
                break;
            case 'p': //p 받으면 출력
                PrintHeap(heap);
                break;
            case 'f': //f 받으면 찾기
                fscanf(fin, "%d", &key);
                Find(heap, key);
                break;
        }
    }
    fclose(fin);
    fclose(fout);
}

void CreateHeap(){ //heap 생성 및 동적 할당
    heap = malloc(sizeof(struct HeapStruct));
    heap->Capacity = fgetc(fin)-'0';
    heap->Elements = malloc(sizeof(int)*heap->Capacity+2);
    heap->Size = 0;
}

void Insert(PriorityQueue h, int value){ //삽입 함수
    int i = 1;
    for (i; i<=h->Size; i++){
        if (value == h->Elements[i]){
            fprintf(fout, "Insert Error: %d is already in the heap\n", value);
            return;
        }
    }
    if (h->Size == h->Capacity){
        fprintf(fout, "Insert Error: heap is full\n");
        return;
    }
    for (i=++(h->Size); h->Elements[i/2]<value && i>=2; i/=2){
        h->Elements[i] = h->Elements[i/2];
    }
    h->Elements[i] = value;
    fprintf(fout, "insert %d\n", value);
}

void Find(PriorityQueue h, int value){ //찾기 함수
    int i = 1;
    for (i; i<h->Size; i++){
        if (value == h->Elements[i]){
            fprintf(fout, "%d is in the heap\n", value);
            return;
        }
    }
    fprintf(fout, "Finding Error: %d is not in the heap\n", value);
}

void DeleteMax(PriorityQueue h){ //삭제 함수
    if (h->Size == 0){
        fprintf(fout, "Delete Error: heap is empty\n");
        return;
    }
    int i=1, child, min_element, max_element;
    min_element = h->Elements[h->Size];
    max_element = h->Elements[1];
    for (i; i*2<=h->Size; i=child){
        child = i*2;
        if (child<h->Size && h->Elements[child+1]>h->Elements[child]){
            child++;
        }
        if (max_element > h->Elements[child]){
            h->Elements[i] = h->Elements[child];
        }
        else break;
    }
    h->Size--;
    fprintf(fout, "Max Element: %d deleted\n", max_element);
}

void PrintHeap(PriorityQueue h){ //출력 함수
    if (h->Size == 0){
        fprintf(fout, "Print Error: heap is empty\n");
        return;
    }
    for (int i=1; i<=h->Size; i++) fprintf(fout, "%d ", h->Elements[i]);
    fprintf(fout, "\n");
}

