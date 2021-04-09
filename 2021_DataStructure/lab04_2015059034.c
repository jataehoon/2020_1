#include<stdio.h>
#include<stdlib.h>

FILE *fin; //파일 읽는 파일 포인터
FILE *fout;//파일 출력하는 파일 포인터

typedef struct ThreadedTree* ThreadedPtr; //포인터 구조체
typedef int ElementType; //Element int 타입 변수

struct ThreadedTree {  //구조체 정의
    int left_thread;   //왼쪽 필드
    ThreadedPtr left_child; //왼쪽 연결 필드
    ElementType data;  //데이터 필드
    ThreadedPtr right_child;//오른쪽 연결 필드
    int right_thread;  //오른쪽 필드
}ThreadedTree;

ThreadedPtr save[100]; //트리 원소 저장할 배열 포인터
ThreadedPtr insucc(ThreadedPtr tree); //inorder successor 찾기
ThreadedPtr CreateTree();//빈 트리 생성
void Insert(ThreadedPtr root, ThreadedPtr save); //트리에 원소 삽입
void printInorder(ThreadedPtr tree); //트리 안의 원소 출력
void DeleteTree(int root_idx, int NUM_NODES); //동적할당 해제 및 트리 삭제

int main(int argc, char *argv[]){
    if (argc==1) {
        fin=fopen("fin.txt", "r");
        fout=fopen("fout.txt", "w");
    }
    else {
        fin=fopen(argv[1], "r");
        fout=fopen(argv[2], "w");
    }

    ThreadedPtr root = CreateTree(); //빈 트리 생성
    int NUM_NODES, root_idx, idx = 0; //필요 변수들 정의
    char checknum; //공백이나 줄내림 예외처리 위한 변수
    fscanf(fin, "%d", &NUM_NODES); //파일에서 첫번째 숫자 읽음

    for (idx; idx < NUM_NODES; ){ //첫번째 숫자만큼 실행
        checknum = fgetc(fin);  //파일에서 이어서 읽어들임
        if (checknum =='\n' || checknum == ' ') continue;//공백 혹은 줄내림이면 스킵
        else idx++;  //조건에 포함 한되면 인덱스 1 증가
        ThreadedPtr root_a = (ThreadedPtr)malloc(sizeof(ThreadedTree));//정의 및 동적할당
        save[idx] = root_a;
        root_a->data = checknum;
        root_a->left_thread = 1;
        root_a->right_thread = 1;
        if (idx == 1) { //인덱스가 1일 경우
            root_a->left_child = root; //루트의 왼쪽 자식을 처음 루트로 정의
            root_a->right_child = root;//루트의 오른쪽 자식을 처음 루트로 정의
            root->left_child = root_a; //기존 루트의 왼쪽 자식을 루트로 정의
        }
        else {  //그 외의 인덱스의 경우
            Insert(root_a, save[idx/2]); //바이너리 트리 원소 삽입
        }
    }

    printInorder(root); //트리 안의 원소 출력
    DeleteTree(root_idx, NUM_NODES); //동적할당 해제 및 트리 삭제

    fclose(fin);
    fclose(fout);

    return 0;
}

ThreadedPtr CreateTree(){ //빈 트리를 생성
    ThreadedPtr tree = (ThreadedPtr)malloc(sizeof(ThreadedTree));
    if (tree==NULL){
        fprintf(fout, "Out of Space!\n");
        return NULL;
    }
    save[0] = tree;
    tree->left_thread = 0;
    tree->right_thread = 0;
    tree->left_child = tree;
    tree->right_child = tree;
    return tree;
}

void Insert(ThreadedPtr root, ThreadedPtr save){ //트리 안에 원소 삽입
    if (save->left_thread) {
        save->left_thread = 0;
        root->left_child = save->left_child;
        root->right_child = save;
        save->left_child = root;
    }
    else {
        save->right_thread = 0;
        root->right_child = save->right_child;
        root->left_child = save;
        save->right_child = root;
    }
}

ThreadedPtr insucc(ThreadedPtr tree) { //inorder predecessor를 찾음
    ThreadedPtr tmp;  //임시 포인터 정의
    tmp = tree->right_child; //임시 포인터가 오른쪽 자식 향함
    if (!tree->right_thread) { //오른쪽이 빈 것이 아닐 경우
        while (!tmp->left_thread) tmp = tmp->left_child; //왼쪽이 빈게 아니면 왼쪽으로 이동
    }
}


void printInorder(ThreadedPtr tree){ //트리 안의 원소들을 출력
    fprintf(fout, "Inorder Traversal :");
    ThreadedPtr tmp = tree;
    for (;;) {
        tmp = insucc(tmp);
        if (tmp == tree) break;
        fprintf(fout, " %c", tmp->data);
    }
    fprintf(fout, "\n");
}

void DeleteTree(int root_idx, int NUM_NODES){ //동적할당 해제 및 트리 삭제
    for (root_idx; root_idx<=NUM_NODES; root_idx++) free(save[root_idx]);
}
