#include<stdio.h>
#include<stdlib.h>

FILE* fin;
FILE* fout;

struct AVLNode;
typedef struct AVLNode* Position;
typedef struct AVLNode* AVLTree;
typedef int ElementType;

typedef struct AVLNode {
    ElementType element;
    AVLTree left, right;
    int height;
}AVLNODE;

AVLTree Insert(ElementType X, AVLTree T);
AVLTree Delete(ElementType X, AVLTree T);
Position SingleRotateWithLeft(Position node);
Position SingleRotateWithRight(Position node);
Position DoubleRotateWithLeft(Position node);
Position DoubleRotateWithRight(Position node);
void PrintInorder(AVLTree T);
void DeleteTree(AVLTree T);
int Height(Position p);
int Height2(Position p);

void main(int argc, char* argv[]){
    if (argc == 1) {
        fin = fopen("input.txt", "r");
        fout = fopen("output.txt", "w");
    }
    else {
        fin = fopen(argv[1], "r");
        fout = fopen(argv[2], "w");
    }
    AVLTree Tree = NULL;
    char cv;
    int key;
    while (!feof(fin)) {
        fscanf(fin, "%c", &cv);
        switch (cv) {
            case 'i': //i를 받으면 삽입
                fscanf(fin, "%d", &key);
                Tree = Insert(key, Tree);
                PrintInorder(Tree);
                fprintf(fout, "\n");
                break;
            case 'd': //d를 받으면 삽입
                fscanf(fin, "%d", &key);
                Delete(key, Tree);
                PrintInorder(Tree);
                fprintf(fout, "\n");
                break;
            }
    }
    DeleteTree(Tree);
    fclose(fin);
    fclose(fout);
}

AVLTree Insert(ElementType X, AVLTree T) { //삽입
    if (T == NULL) {
        T = (AVLTree)malloc(sizeof(struct AVLNode));
        if (T != NULL) {
            T->element = X;
            T->height = 0;
            T->left = T->right = NULL;
        }
    }
    else if (T->element < X) {
        T->right = Insert(X, T->right);
        if (Height(T->right) - Height(T->left) == 2) {
            if (X > T->right->element) T = SingleRotateWithRight(T);
            else T = DoubleRotateWithRight(T);
        }
    }
    else if (T->element > X) {
        T->left = Insert(X, T->left);
        if (Height(T->left) - Height(T->right) == 2) {
            if (X < T->left->element) T = SingleRotateWithLeft(T);
            else T = DoubleRotateWithLeft(T);
        }
    }
    else fprintf(fout, "Insertion Error: %d is already in the tree!\n", X);
    T->height = Height2(T);
    return T;
}

AVLTree Delete(ElementType X, AVLTree T) {  //삭제
    AVLTree temp;  //임시 트리
    if (T == NULL) fprintf(fout, "Deletion Error: %d is not in the tree!\n", X);
    else if (T->element < X) T->right = Delete(X, T->right);
    else if (T->element > X) T->left = Delete(X, T->left);
    else if (T->left != NULL && T->right != NULL) { //자식 둘일때
        if (T->left == NULL) temp = NULL;
        else temp = T->left;
        T->element = temp->element;
        T->left = Delete(T->element, T->left);
    }
    else {
        temp = T;
        if (T->right == NULL) T = T->left;
        else if (T->left == NULL) T = T->right;
        free(temp);
    }
    return T;
}

Position SingleRotateWithLeft(Position node) {
    Position tmp = node->left;
    node->left = tmp->right;
    tmp->right = node;
    node->height = Height2(node);
    tmp->height = Height2(tmp);
    return tmp;
}

Position SingleRotateWithRight(Position node) {
    Position tmp = node->right;
    node->right = tmp->left;
    tmp->left = node;
    node->height = Height2(node);
    tmp->height = Height2(tmp);
    return tmp;
}

Position DoubleRotateWithLeft(Position node) {
    node->left = SingleRotateWithRight(node->left);
    return SingleRotateWithLeft(node);
}

Position DoubleRotateWithRight(Position node) {
    node->right = SingleRotateWithLeft(node->right);
    return SingleRotateWithRight(node);
}

void PrintInorder(AVLTree T) {
    if (T == NULL) return;
    PrintInorder(T->left);
    fprintf(fout, "%d(%d) ", T->element, Height(T));
    PrintInorder(T->right);
}

void DeleteTree(AVLTree T){
    free(T);
}

int Height(Position p) {
    if (p == NULL) return -1;
    else return p->height;
}

int Height2(Position p) {
    if (Height(p->left)>Height(p->right)) return Height(p->left)+1;
    else return Height(p->right)+1;
}
