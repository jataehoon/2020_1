#include <stdio.h>

int main()
{
    int size;
    printf("리스트의 크기: ");
    scanf("%d", &size);
    int list[size];
    for (int i=0; i<size; i++) {
        printf("%d번째 원소: ", i+1);
        scanf("%d", &list[i]);
    }
    printf("\n");
    int temp = list[0];
    for (int i=0; i<size; i++) {
        for (int j=0; j<size-1; j++) {
            if (list[j]>list[j+1]) {
                temp = list[j];
                list[j] = list[j+1];
                list[j+1] = temp;
            }
        }
    }
    printf("정렬된 리스트: [");
    for (int i=0; i<size; i++) {
        if (i==size-1) printf("%d", list[i]);
        else printf("%d, ", list[i]);
    }
    printf("]\n");
}
