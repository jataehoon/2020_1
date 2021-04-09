#include<stdio.h>
#include<stdlib.h>

typedef struct Node *PtrToNode;
typedef PtrToNode List;
typedef PtrToNode Position;
typedef int ElementType;

struct Node   //구조체 정의
{
    ElementType element;
    Position next;
};

FILE *input;
FILE *output;

List MakeEmpty() {    //빈 리스트를 만든다
    List list;   //포인터 형식의 리스트 선언
    list = (List) malloc(sizeof(struct Node));  //리스트에 동적할당
    list->element = 0;  //리스트의 원소를 임의의 값으로 설정
    list->next = NULL;  //list->next is NULL
    return list;  //빈 리스트를 나타내는 포인터 반환
}

int IsEmpty(List list) {   //리스트가 비어있는지 확인
    return list->next == NULL; //비어있으면 1, 아니면 0 반환
}

int IsLast(Position p, List list) {  //해당 위치가 마지막인지 확인
    return p->next == NULL;  //맞으면 1, 아니면 0 반환
}

void Insert(ElementType x, List list, Position p) {  //리스트에 원소를 집어넣는다
    Position tmp;   //임시 포인터 정의
    tmp = (Position) malloc(sizeof(struct Node));   //tmp에 동적할당
    tmp->element = x;    //tmp의 원소는 x이다
    tmp->next = p->next;   //연결 끊고 다음 리스트에 연결
    p->next = tmp;    //꼬리 부분을 이전 리스트와 연결
}

void PrintList(List list) {    //리스트를 출력
    Position p;   //포인터 p를 선언
    p = list->next;  //p를 다음과 같이 정의
    if (IsEmpty(list)) fprintf(output, "Empty List!\n");  //리스트가 비어있을 경우 에러
    while (p != NULL) {  //p가 널값이 아닐 경우
        fprintf(output, "Key: %d\t", p->element);  //각 원소를 출력
        p = p->next;  //다음 원소로 넘어간다
    }
    fprintf(output, "\n");
}

Position Find(ElementType x, List list) {   //원소를 찾는다
    Position p;  //포인터 p를 선언
    p = list->next;  //p를 다음과 같이 정의
    while (p != NULL && p->element != x) {  //p가 널이 아니고 원소가 x가 아닐때
        p = p->next;  //다음 원소로 넘어간다
    }
    return p;  //포인터 p 반환 
}

Position FindPrevious(ElementType x, List list) {  //이전 원소를 찾는다
    Position p;  //포인터 p 선언
    p = list;   //p를 다음과 같이 정의
    while (p->next != NULL && p->next->element != x) {  //다음 조건 하에
        p = p->next;  //다음 원소로 넘어간다
    }
    return p;   //포인터 p 반환
}

void Delete(ElementType x, List list) {  //원소를 지운다
    Position p, tmp;  //포인터 2개 선언
    p = FindPrevious(x, list);  //FindPrevious함수 이용
    if (!IsLast(p, list)) {  //p가 마지막 원소가 아니라면
        tmp = p->next;  //다음과 같이 정의
        p->next = tmp->next; //경로 변경하여 지우는 방식
        free(tmp);   //할당된 tmp를 해제
    }
}

void DeleteList(List list) { //리스트를 지운다
    Position p, tmp;  //포인터 2개 선언
    p = list->next;   //다음과 같이 p 정의
    list->next = NULL;  //다음과 같이 정의
    while (p != NULL) {  //p가 널이 아니라면
        tmp = p->next;  //다음과 같이 정의
        free(p);  //할당된 p 해제
        p = tmp;  //p를 tmp로 둔다
    }
}

int main(int argc, char *argv[]){
    char command;
    int key1, key2;
    FILE *input;
    Position header = NULL;
    Position tmp = NULL;
    if (argc == 1) {
        input = fopen("input.txt", "r");
        output = fopen("output.txt", "w");
    }
    else {
        input = fopen(argv[1], "r");
        output = fopen(argv[2], "w");
    }
    header = MakeEmpty(header);
    while (1) {
        command = fgetc(input);
        if (feof(input)) break;
        switch (command) {
            case 'i':
                fscanf(input, "%d %d", &key1, &key2);
                //duplication check
                tmp = Find(key1, header);
                if (tmp != NULL) { //삽입하려는 키가 이미 리스트에 존재시 에러 발생
                    fprintf(output, "Insertion(%d) Failed: cannot find the location to be inserted\n", key1);
                    break;
                }
                tmp = Find(key2, header);
                if (key2 != -1 && tmp == NULL) { //키2가 -1이 아닌데 리스트에 없다면 에러 발생
                    fprintf(output, "Insertion(%d) Failed: cannot find the location to be inserted\n", key1);
                    break;
                }
                if (key2 == -1) tmp=header; //키2가 -1이면 헤더 다음에 삽입될 수 있도록 tmp로 헤더를 가르킴
                Insert(key1, header, tmp);
                break;
            case 'd':
                fscanf(input, "%d", &key1);
                tmp = Find(key1, header);
                if (tmp == NULL) {  //tmp가 널일때 key1을 찾을수 없다
                    fprintf(output, "Deletion(%d) Failed: Element %d is not in the list\n", key1, key1);
                    break;
                }
                Delete(key1, header);
                break;
            case 'f':
                fscanf(input, "%d", &key1);
                tmp = FindPrevious(key1, header);
                if (IsLast(tmp, header)) {
                    fprintf(output, "Finding(%d) Failed: Element %d is not in the list\n", key1, key1);
                }
                else {
                    if (tmp->element > 0)
                        fprintf(output, "Key of the previous node of %d is %d\n", key1, tmp->element);
                    else
                        fprintf(output, "Key of the previous node of %d is head\n", key1);
                }
                break;
            case 'p':
                PrintList(header);
                break;
            default:
                break;
        }
    }
    
    DeleteList(header);
    fclose(input);
    fclose(output);
    return 0;
}
