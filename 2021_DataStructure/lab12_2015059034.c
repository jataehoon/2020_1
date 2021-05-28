#include<stdio.h>
#include<stdlib.h>
#define INF 1e9

FILE *fin;
FILE *fout;

typedef struct Node {
    int key;
    int weight;
}Node;

typedef struct Graph {
    int size;
    int** vertices;
    Node* nodes;
}Graph;

typedef struct Heap {
    int capacity;
    int size;
    Node* element;
}Heap;

int d[1000];
Graph createGraph(int size);
void deleteGraph(Graph g);
void printShortestPath(Graph g);
Heap* createMinHeap(int heapSize);
void deleteMinHeap(Graph g, int size);
void insertToMinHeap(Heap* minHeap, int vertex, int distance);

int main(int argc, char *argv[]){
    if (argc == 1) {
        fin = fopen("input.txt", "r");
        fout = fopen("output.txt", "w");
    }
    else {
        fin = fopen(argv[1], "r");
        fout = fopen(argv[2], "w");
    }
    Graph g;
    int size, k=0, sum=0;
    fscanf(fin, "%d\n", &size);
    g = createGraph(size + 1);
    char tmp = 0;
    while (tmp != '\n' && tmp != EOF) {
        int node1, node2, weight;
        fscanf(fin, "%d-%d-%d", &node1, &node2, &weight);
        g.vertices[node1][node2] = weight;
        for (int i = 0; i < k; i++) if (d[i] == node2) {
            fprintf(fout, "Cannot reach to node %d\n", k+2);
            goto loop;
        }
        d[k++] = node1; d[k] = node2;
        sum += weight;
        for (int i = 0; i < k; i++) fprintf(fout, "%d->", d[i]);
        fprintf(fout, "%d (cost: %d)\n", node2, sum);
loop:
        tmp = fgetc(fin);
    }
    deleteMinHeap(g, size);
}

Graph createGraph(int sizes){
    Graph G;
    G.size = sizes;
    G.vertices = malloc(sizeof(Graph)*sizes);
    for (int i = 0; i < sizes-1; i++) {
        G.vertices[i] = malloc(sizeof(Graph)*sizes);
    }
    for (int i = 0; i < sizes-1; i++) {
        for (int j = 0; j < sizes-1; j++) G.vertices[i][j] = 0;
    }
    return G;
}

void deleteMinHeap(Graph G, int sizes) {
    for (int i = 0; i < sizes-1; i++) free(G.vertices[i]);
    free(G.vertices);
}

Heap* createMinHeap(int heapSize) {
    Heap* hp = malloc(sizeof(Heap));
    hp->capacity = INF;
    hp->size = 0;
    hp->element = malloc(sizeof(Node)*INF+1);
    hp->element[0].weight = 0;
    return hp;
}

void insertToMinHeap(Heap* minHeap, int vertex, int distance) {
    if (minHeap->size == minHeap->capacity) return;
    int i;
    for (i=++minHeap->size; minHeap->element[i/2].weight>distance; i/=2){
        minHeap->element[i].key = minHeap->element[i/2].key;
        minHeap->element[i].weight = minHeap->element[i/2].weight;
    }
    minHeap->element[i].key = vertex;
    minHeap->element[i].weight = distance;
}
