#include<stdio.h>
#include<stdlib.h>

FILE *fin;
FILE *fout;

typedef struct BST* Tree;
typedef struct BST{
    int value;
    struct BST* left;
    struct BST* right;
}BST;

Tree insertNode(Tree root, int key);
Tree deleteNode(Tree root, int key);
int findNode(Tree root, int key);
void printInorder(Tree root);
void deleteTree(Tree root);

void main(int argc, char* argv[]){
    if (argc == 1) {
        fin = fopen("input.txt", "r");
        fout = fopen("output.txt", "w");
    }
    else {
        fin = fopen(argv[1], "r");
        fout = fopen(argv[2], "w");
    }
    char cv;
    int key;
    Tree root = NULL;
    while(!feof(fin)){
        fscanf(fin, "%c", &cv);
        switch(cv){
            case 'i': //i를 받을 경우, 삽입
                fscanf(fin, "%d", &key);
                fprintf(fout, "insert %d\n", key);
                if (root == NULL) {
                    root = (Tree)malloc(sizeof(BST));
                    root->value = key;
                    root->left = root->right = NULL;
                    continue;
                }
                insertNode(root, key);
                break;
            case 'f': //f를 받을 경우, 검색
                fscanf(fin, "%d", &key);
                int tmp = findNode(root, key);
                if (tmp) fprintf(fout, "%d is in the tree\n", key);
                else fprintf(fout, "Finding Error: %d is not in the tree\n", key);
                break;
            case 'd': //d를 받을 경우, 삭제
                fscanf(fin, "%d", &key);
                fprintf(fout, "delete %d\n", key);
                deleteNode(root, key);
                break;
            case 'p': //p를 받을 경우, 출력
                fscanf(fin, "%c", &cv);
                if (cv == 'i'){
                    if (root == NULL) fprintf(fout, "Tree is empty\n");
                    else printInorder(root);
                }
                fprintf(fout, "\n");
        }
    }
    deleteTree(root);
}

Tree insertNode(Tree root, int key){  //트리 안에 원소 삽입
    if (root == NULL){  //트리가 비어있을 경우
        root = (Tree)malloc(sizeof(BST));
        root->value = key;
        root->left = root->right = NULL;
    }
    else if (root->value < key) { //키값이 트리의 값보다 클 경우
        root->right = insertNode(root->right, key);
    }
    else if (root->value > key) { //키값이 트리의 값보다 작을경우
        root->left = insertNode(root->left, key);
    }
    else {
        fprintf(fout, "Insertion Error: %d is already in the tree\n", key);
    }
    return root; //트리 반환
}

Tree deleteNode(Tree root, int key){  //트리의 원소 삭제
    Tree temp; //임시 트리형 포인터
    if (root == NULL) {  //트리가 해당 원소에 없을때 오류 출력
        fprintf(fout, "Deletion Error: %d is not in the tree\n", key);
    }
    else if (root->value < key) { //키값이 트리의 값보다 클 경우
        root->right = deleteNode(root->right, key);
    }
    else if (root->value > key) { //키값이 트리의 값보다 작을경우
        root->left = deleteNode(root->left, key);
    }
    else if (root->left != NULL && root->right != NULL) { //자식 둘일때
        if (root->left == NULL) temp = NULL;
        else {
            while (root->left->right != NULL) root->left->right;
            temp = root->left;
        }
        root->value = temp->value;
        root->left = deleteNode(root->left, root->value);
    }
    else {
        temp = root;
        if (root->right == NULL) root = root->left; //오른쪽이 널이면 왼쪽
        else if (root->left == NULL) root = root->right; //왼쪽 널이면 오른쪽
        free(temp);
    }
    return root;
}

int findNode(Tree root, int key){  //트리의 원소 찾기
    if (root == NULL) return 0;
    if (root->value < key) return findNode(root->right, key);
    else if (root->value > key) return findNode(root->left, key);
    else return 1;
}

void printInorder(Tree root){   //트리의 원소 inorder로 출력
    if (root == NULL) return;
    printInorder(root->left);
    fprintf(fout, "%d ", root->value);
    printInorder(root->right);
}

void deleteTree(Tree root){  //트리 삭제
    free(root);
}
