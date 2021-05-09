#include<stdio.h>
#include<stdlib.h>

FILE *fin;
FILE *fout;

typedef struct BNode{
    int order;
    int key[2];
    struct BNode* child[3];
}BNode;

void CreateTree(BNode* root, int key); //트리 생성
int Insert(int key); //비트리에 삽입
void Find(BNode* root, int key); //비트리 검색
void PrintTree(BNode* root); //비트리 원소 출력
int Push(BNode* stack[], BNode* left, BNode* right, int i, int j);
int find_helper;
BNode* root;

int main(int argc, char* argv[]){
    if (argc == 1){
        fin = fopen("input.txt", "r");
        fout = fopen("output.txt", "w");
    }
    else {
        fin = fopen(argv[1], "r");
        fout = fopen(argv[2], "w");
    }
    root = NULL;
    char cv;
    int key;
    while (!feof(fin)){
        cv = fgetc(fin);
        switch (cv) {
            case 'i':
                fscanf(fin, "%d", &key);
                Insert(key);
                break;
            case 'f':
                find_helper = 0;
                fscanf(fin, "%d", &key);
                Find(root, key);
                if (find_helper != 0) fprintf(fout, "Find key %d\n", key);
                else fprintf(fout, "Finding Error: key %d is not in the tree!\n", key);
                break;
            case 'p':
                if (root == NULL) {
                    fprintf(fout, "Print Error: tree is empty!\n");
                    break;
                }
                PrintTree(root);
                fprintf(fout, "\n");
                break;
        }
    }
}

void CreateTree(BNode* b, int k) { //빈 비트리 생성
    b = (BNode*)malloc(sizeof(BNode));
    b->order = 1;
    b->key[0] = k;
    b->key[1] = 0;
    b->child[0] = NULL; //차일드가 모두 널을 가르키도록 설정
    b->child[1] = NULL;
    b->child[2] = NULL;
    root = b;
}

int Insert(int k){
    BNode* stack[100]; //stack의 위치에 k의 key값으로 삽입
    int i = 0;
    BNode* b = root; //b가 현재 노드 가르키도록 설정
    if (b == NULL) { //root가 널이면 root 새로 생성
        CreateTree(b,k);
        return 1;
    }
    while (1){ //스택을 확인하여 삽입될 위치를 찾는다
        stack[i] = b;
        if (k < b->key[0]) b = b->child[0];
        else if (k == b->key[0]){ //해당 키값이 이미 존재할 경우 에러 출력
            fprintf(fout, "Insertion Error: key %d is already in the tree!\n", k);
            return 0; //에러 출력 및 강제 종료
        }
        else if (b->order == 1 || k<b->key[1]) b = b->child[1];
        else if (k>b->key[1]) b = b->child[2];
        else return 0;
        i++;
        if (b == NULL) {
            b = stack[--i]; //b는 삽입할 노드, stack[i]의 위치에 삽입
            break;
        }
    }
    if (Push(stack, NULL, NULL, i, k)==1) return 1;
    else return 0;
}

void Find(BNode* b, int k){ //비트리 탐색 함수
    if (b != NULL) {
        Find(b->child[0], k); //재귀
        if (k == b->key[0]) find_helper++; //재귀로 찾아서 해당변수+1
        Find(b->child[1], k); //재귀
        if (b->order == 2){
            if (k == b->key[0]) find_helper++; //해당변수+1
            Find(b->child[2], k); //재귀
        }
    }
}

void PrintTree(BNode* b){ //비트리 원소 출력
    if (b != NULL) {
        PrintTree(b->child[0]); //재귀
        fprintf(fout, "%d ", b->key[0]);
        PrintTree(b->child[1]); //재귀
        if (b->order == 2) {
            fprintf(fout, "%d ", b->key[1]);
            PrintTree(b->child[2]); //재귀
        }
    }
}

int Push(BNode* stack[], BNode* left, BNode* right, int i, int j){
    BNode* b;
    if (i<0){ //root가 꽉찬 상태에서 분할시 높이 1 증가
        root = (BNode*)malloc(sizeof(BNode));
        root->order = 1;
        root->key[0] = j;
        root->child[0] = left;
        root->child[1] = right;
        return 1;
    }
    b = stack[i]; //b는 현재 노드를 가르킴
    if (b->order == 1){ //삽입할 곳의 오더가 1이면 정상적으로 삽입
        if (b->key[0] < j){
            b->key[1] = j;
            b->child[1] = left;
            b->child[2] = right;
        }
        else {
            b->key[1] = b->key[0];
            b->key[0] = j;
            b->child[2] = b->child[1];
            b->child[1] = right;
            b->child[0] = left;
        }
        b->order = 2;
        return 1;
    }
    else if (b->order == 2){ //삽입될 곳의 오더가 2면 재귀 돌린다
        BNode* LEFT, * RIGHT; //재귀할 때 올려줄 left right 정의
        int KEY;
        RIGHT = (BNode*)malloc(sizeof(BNode));
        RIGHT->order = 1;
        LEFT = b;
        LEFT->order = 1;
        if (j<b->key[0]){
            KEY = b->key[0];
            RIGHT->key[0] = b->key[1];
            RIGHT->child[0] = b->child[1];
            RIGHT->child[1] = b->child[2];
            LEFT->key[0] = j;
            LEFT->child[0] = left;
            LEFT->child[1] = right;
        }
        else if (j<b->key[1]){
            KEY = j;
            RIGHT->key[0] = b->key[1];
            RIGHT->child[0] = right;
            RIGHT->child[1] = b->child[2];
            LEFT->child[1] = left;
        }
        else {
            KEY = b->key[1];
            RIGHT->key[0] = j;
            RIGHT->child[0] = left;
            RIGHT->child[1] = right;
        }
        if (Push(stack, LEFT, RIGHT, i-1, KEY)==0) return 1;
        else return 0;
    }
    return 0;
}

