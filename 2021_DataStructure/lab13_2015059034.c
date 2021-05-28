#include<stdio.h>
#include<stdlib.h>
#include<string.h>

FILE *fin;
FILE *fout;

typedef int ElementType;
typedef struct LNode* Position;
typedef Position List;
typedef struct HashTbl* HashTable;

struct HashTbl {
    int TableSize;
    List TheLists;
};

struct LNode {
    int element;
};

HashTable createTable(int TableSize); //해시테이블 생성
void Insert(HashTable H, ElementType Key, int solution);//삽입
void Delete(HashTable H, ElementType Key, int solution);//삭제
int Find(HashTable H, ElementType Key, int solution);//검색
void printTable(HashTable H); //테이블 원소 출력
void deleteTable(HashTable H); //동적할당 해제 및 테이블 삭제

int main(int argc, char* argv[]){
    if (argc == 1){
        fin = fopen("input.txt", "r");
        fout = fopen("output.txt", "w");
    }
    else {
        fin = fopen(argv[1], "r");
        fout = fopen(argv[2], "w");
    }
    char solution_str[20];//1차 2차 구분
    int solution, TableSize;
    fscanf(fin, "%s", solution_str);
    if (!strcmp(solution_str, "Linear")) solution = 1;
    else if (!strcmp(solution_str, "Quadratic")) solution = 2;
    else {
        fprintf(fout, "Error: Invalid hashing solution!\n");
        return 0;
    }
    fscanf(fin, "%d", &TableSize); //테이블의 크기
    HashTable H = createTable(TableSize);
    char cv;
    int key;
    while (!feof(fin)) {
        fscanf(fin, "%c", &cv);
        switch (cv) {
            case 'i': //삽입
                fscanf(fin, "%d", &key);
                Insert(H, key, solution);
                break;
            case 'f': //검색
                fscanf(fin, "%d", &key);
                if (Find(H, key, solution)>=0) fprintf(fout, "%d is in the table\n", key);
                else fprintf(fout, "%d is not in the table\n", key);
                break;
            case 'd': //삭제
                fscanf(fin, "%d", &key);
                Delete(H, key, solution);
                break;
            case 'p': //출력
                //fscanf(fin, "%c", &cv);
                printTable(H);
                break;
        }
    }
    deleteTable(H); //해쉬 테이블 해제
    fclose(fin);
    fclose(fout);
    return 0;
}

int Hash(int value, int TableSize, int solution, int i){ //해시값 반환 함수
    if (solution == 1) value += i; //linear의 경우
    else if (solution == 2) value += i*i; //quadratic의 경우
    return (value % TableSize);
}

HashTable createTable(int TableSize) { //해시테이블 생성
    HashTable New = malloc(sizeof(struct HashTbl));
    New->TableSize = TableSize;
    New->TheLists = malloc(sizeof(struct LNode)*TableSize);
    for (int i=0; i<TableSize; i++) New->TheLists[i].element = 0;
    return New;
}

void Insert(HashTable H, ElementType Key, int solution) { //삽입
    if (Find(H, Key, solution) < 0) {
        int cursor, ind, i = 0;
        cursor = ind = Hash(Key, H->TableSize, solution, i++);
        while (H->TheLists[ind].element) {
            ind = Hash(Key, H->TableSize, solution, i++);
            if (cursor == ind) {
                fprintf(fout, "Insertion Error: table is full\n");
                return;
            }
        }
        H->TheLists[ind].element = Key;
        fprintf(fout, "Insert %d into address %d\n", Key, ind);
    }
    else {
        int i = 0;
        while (H->TheLists[i].element != Key) i++;
        fprintf(fout, "Insertion Error: %d already exists at address %d\n", Key, i);
    }
}

void Delete(HashTable H, ElementType Key, int solution) { //삭제
    int ind = Find(H, Key, solution);
    if (ind >= 0) {
        H->TheLists[ind].element = 0;
        fprintf(fout, "Delete %d\n", Key);
    }
    else fprintf(fout, "Deletion Error: %d is not in the table\n", Key);
}

int Find(HashTable H, ElementType Key, int solution) { //검색
    int cursor, ind, i = 0;
    cursor = ind = Hash(Key, H->TableSize, solution, i++);
    while (H->TheLists[ind].element != Key) {
        if (H->TheLists[ind].element == 0) {
            ind = -1;
            break;
        }
        ind = Hash(Key, H->TableSize, solution, i++);
        if (H->TheLists[ind].element == cursor) {
            ind = -1;
            break;
        }
    }
    return ind;
}

void printTable(HashTable H) { //테이블 원소 출력
    for (int i=0; i<H->TableSize; i++) fprintf(fout, "%d ", H->TheLists[i].element);
    fprintf(fout, "\n");
}

void deleteTable(HashTable H) { //테이블 해제 및 삭제
    free(H->TheLists);
    free(H);
}
