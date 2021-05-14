#include<stdio.h>
#include<stdlib.h>

FILE *fin;
FILE *fout;

typedef struct _Queue {
    int *key;
    int first, rear, qsize, max_queue_size;
}Queue;

typedef struct _Graph {
    int size;
    int* node;
    int** matrix;
}Graph;

Graph CreateGraph(int* nodes); //node n개인 빈 그래프 생성
void InsertEdge(Graph G, int a, int b); //node a에서 b로가는 edge생성
void PrintGraph(Graph G); //그래프를 출력
void DeleteGraph(Graph G); //그래프 동적할당 해제 및 삭제
void Topsort(Graph G); //위상 정렬
Queue MakeNewQueue(int X); //x의 크기의 큐를 생성
void Enqueue(Queue* Q, int X); //큐의 원소를 팝
int Dequeue(Queue* Q); //큐의 끝에 원소추가
int cnt = 0; //인접행렬 원소 저장 위해 전역변수로 관리

int main(int argc, char* argv[]){
    if (argc == 1) {
        fin = fopen("input.txt", "r");
        fout = fopen("output.txt", "w");
    }
    else {
        fin = fopen(argv[1], "r");
        fout = fopen(argv[2], "w");
    }
    int* arr; //인접행렬
    int a, b, tmp_asc; //노드 a,b,오름정렬 위한 변수
    char n = ' '; //입력받을 문자 공백으로 초기화
    while (n != '\n') { //엔터 입력 받을때까지 반복
        fscanf(fin, "%d", &a); //숫자 받음
        fscanf(fin, "%c", &n); //문자 받음
        if (!cnt) arr = (int*)malloc(sizeof(int)*(++cnt));
        else arr = (int*)realloc(arr, sizeof(int)*(++cnt));
        arr[cnt-1] = a;
    }
    for (int i=0; i < cnt; i++) { //오름 정렬
        for (int j=0; j < cnt-i; j++) {
            if (arr[j] > arr[j+1]) {
                tmp_asc = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = tmp_asc;
            }
        }
    }
    Graph G = CreateGraph(arr); //빈 그래프 생성
    while (fscanf(fin, "%d", &a) != EOF) {
        fscanf(fin, "%c", &n);
        if (n == '-') { //문자에 -를 받을시 삽입
            fscanf(fin, "%d", &b);
            InsertEdge(G, a, b);
        }
        fscanf(fin, "%c", &n);
    }
    PrintGraph(G); //그래프 출력
    fprintf(fout, "\n"); //공백 1줄
    Topsort(G); //위상 정렬
    DeleteGraph(G); //그래프 동적할당 해제 및 삭제
}

Graph CreateGraph(int* nodes) { //node n개인 빈 그래프 생성
    Graph G;
    G.size = cnt;
    G.node = nodes;
    G.matrix = malloc(sizeof(int*)*(cnt+1));
    for (int i=0; i < cnt+1; i++) {
        G.matrix[i] = malloc(sizeof(int)*(cnt+1));
    }
    for (int i=0; i < cnt; i++) {
        for (int j=0; j < cnt; j++) G.matrix[i][j] = 0;
    }
    return G;
}

void InsertEdge(Graph G, int a, int b) { //node a에서 b로가는 edge삽입
    int x,y;
    for (int i=0; i < cnt; i++) {
        if (G.node[i] == a) x = i;
        if (G.node[i] == b) y = i;
    }
    G.matrix[x][y] = 1;
}

void PrintGraph(Graph G) { //그래프를 출력
    for (int i=0; i < cnt; i++) {
        for (int j=0; j < cnt; j++) {
            if (G.matrix[i][j]==G.matrix[j][i]&&G.matrix[i][j]==1){
                fprintf(fout, "Sorting Error, Cycle!\n");
                return;
            }
        }
    }
    for (int i=0; i < cnt+1; i++) {
        for (int j=0; j < cnt+1; j++) {
            if (i==0 && j==0) fprintf(fout, "   ");
            else if (i==0) fprintf(fout, "%d  ", G.node[j-1]);
            else if (j==0) fprintf(fout, "%d  ", G.node[i-1]);
            else fprintf(fout, "%d  ", G.matrix[i-1][j-1]);
        }
        fprintf(fout, "\n");
    }
}

void DeleteGraph(Graph G) { //그래프 동적할당 해제 및 삭제
    free(G.node);
    for (int i=0; i < cnt+1; i++) free(G.matrix[i]);
    free(G.matrix);
}

void Topsort(Graph G) { //위상 정렬
    for (int i=0; i < cnt; i++) {
        for (int j=0; j < cnt; j++) {
            if (G.matrix[i][j]==G.matrix[j][i]&&G.matrix[i][j]==1){
                fprintf(fout, "Sorting Error: Cycle!\n");
                return;
            }
        }
    }
    Queue Q1;
    Q1.qsize = 0;
    int *list = malloc(sizeof(int)*cnt);
    for (int i=0; i < cnt; i++) {
        for (int j=0; j < cnt; j++) if (G.matrix[i][j]) list[j]++;
    }
    for (int i=0; i < cnt; i++) {
        if (!list[i] && !Q1.qsize) Q1 = MakeNewQueue(i);
        else if (!list[i]) Enqueue(&Q1, i);
    }
    while (Q1.qsize) {
        int* top = malloc(sizeof(int)*Q1.qsize);
        int q1 = Q1.qsize;
        for (int i=0; i<q1; i++) top[i] = Dequeue(&Q1);
        for (int i=0; i<q1; i++) {
            for (int j=0; j<q1; j++) {
                if (G.node[top[i]] < G.node[top[j]]) {
                    int tmp = top[i];
                    top[i] = top[j];
                    top[j] = tmp;
                }
            }
        }
        for (int i=0; i<q1; i++) fprintf(fout, "%d ", G.node[top[i]]);
        for (int i=0; i<q1; i++) {
            for (int j=0; j<cnt; j++) {
                if (G.matrix[top[i]][j]) {
                    list[j]--;
                    if (!list[j] && Q1.qsize) Enqueue(&Q1, j);
                    else if (!list[j]) Q1 = MakeNewQueue(j);
                }
            }
        }
        free(top);
    }
    fprintf(fout, "\n");
}

Queue MakeNewQueue(int X) { //x의 크기의 큐를 생성
    Queue Q2;
    Q2.max_queue_size = cnt;
    Q2.first = Q2.rear = X;
    Q2.key = malloc(sizeof(int));
    Q2.key[0] = X;
    Q2.qsize = 1;
    return Q2;
}

void Enqueue(Queue* Q, int X) { //큐의 원소를 팝
    if (Q->qsize == Q->max_queue_size) return;
    Q->key = realloc(Q->key, (++Q->qsize)*sizeof(int));
    Q->key[Q->qsize-1] = X;
    Q->rear = X;
}

int Dequeue(Queue* Q) { //큐의 끝에 원소추가
    if (!(Q->qsize)) return 99999999;
    int q1st = Q->first;
    int* tmp_q = malloc(sizeof(int)*(--Q->qsize));
    for (int i=1; i < Q->qsize+1; i++) tmp_q[i-1] = Q->key[i];
    Q->key = tmp_q;
    if (Q->qsize) Q->first = Q->key[0];
    return q1st;
}
