#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#define ISFULL 1
#define ISEMPTY 2
#define DIVIDEBYZERO 3

FILE *fin; //파일 입출력
FILE *fout;//파일 입출력

typedef struct Stack{ //구조체 정의
    int *key;
    int top;
    int max_stack_size;
}Stack;

Stack* CreateStack(int max){ //스택을 최초에 만든다
    Stack* S = NULL; //널이라고 둔다
    S = (Stack*)malloc(sizeof(Stack)); //스택 S에 동적 할당
    S->key = (int*)malloc(max*sizeof(int)); //S의 key에 동적 할당
    S->max_stack_size = max; //최대 사이즈
    S->top = -1; //s의 맨 위 원소는 없다 즉 -1로 표시한다
    return S; //s를 반환
}

int IsEmpty(Stack* S){ //스택이 비어있는지 확인
    return S->top == -1;
}
 
int IsFull(Stack* S){ //스택이 꽉 차있는지 확인
    return S->top == 20;
}

void Push(Stack* S, int x){ //스택에 푸쉬
    if (IsFull(S)) { //풀스택의 경우 에러 출력
        fprintf(fout, "\nError : Invalid Postfix Expression, Stack is Full!\n");
        free(S); //메모리 할당 반환
        exit(1); //강제종료
    }
    else {  //나머지의 경우 원소를 푸쉬
        S->key[++(S->top)] = x;
    }
}

int Pop(Stack* S){ //스택에서 팝
    if (IsEmpty(S)) { //스택이 비어있을 경우 에러 출력
        fprintf(fout, "\nError : Invalid Postfix Expression, Stack is Empty!\n");
        free(S); //메모리 할당 반환
        exit(1); //강제종료
    }
    else {  //나머지의 경우 원소를 팝
        return S->key[(S->top--)];
    }
}

int Top(Stack* S){  //스택의 최상위 원소
    if (!IsEmpty(S)) { //스택이 비어있지 않다면
        return S->key[S->top]; //맨 위의 원소를 반환
    }
    else { //그렇지 않은 경우 에러 출력
        fprintf(fout, "\nError! Stack is empty\n");
        free(S); //메모리 할당 반환
        exit(1); //강제종료
    }
}

void DeleteStack(Stack* S){  //동적할당 메모리 반환
    free(S);
}

void main(int argc, char *argv[]){
    if (argc==1) {
        fin = fopen("fin.txt", "r");
        fout = fopen("fout.txt", "w");
    }
    else {
        fin = fopen(argv[1], "r");
        fout = fopen(argv[2], "w");
    }
    Stack* stack;
    char input_str[101];
    int max=20, i=0,a,b,result;
    int error_flag=0;

    fgets(input_str,101,fin);
    stack = CreateStack(max);

    fprintf(fout, "Top Numbers :");
    while (input_str[i]!='#'){ //텍스트 파일에서 # 만날때까지 반복
        switch(input_str[i]){  //텍스트 파일이 길이만큼 실행
            case '+':  //덧셈이 나올 경우
                a = Pop(stack)-'0'; //0 뺌으로 정수로 전환
                b = Pop(stack)-'0'; //0 뺌으로 정수로 전환
                result = a+b; //덧셈 정의
                fprintf(fout, " %d", result); //더한값 출력
                Push(stack, result+'0'); //해당 원소 푸쉬
                error_flag--; //스택 안의 원소 갯수
                break;
            case '-':
                a = Pop(stack)-'0';
                b = Pop(stack)-'0';
                result = b-a; //뺄셈 정의
                fprintf(fout, " %d", result); //뺀값 출력
                Push(stack, result+'0'); //해당 원소 푸쉬
                error_flag--;
                break;
            case '*':
                a = Pop(stack)-'0';
                b = Pop(stack)-'0';
                result = b*a; //곱셈 정의
                fprintf(fout, " %d", result); //곱한값 출력
                Push(stack, result+'0'); //해당 원소 푸쉬
                error_flag--; //스택 안의 원소 갯수
                break;
            case '/':
                a = Pop(stack)-'0';
                b = Pop(stack)-'0';
                if (a == 0) { //분모가 0일 경우 에러 출력
                    fprintf(fout, "\nError : Invalid Postfix Expression, Divide by Zero!\n");
                    DeleteStack(stack); //메모리 반환
                    exit(1); //강제종료
                }
                result = b/a; //나눗셈 정의
                fprintf(fout, " %d", result); //나눈값 출력
                Push(stack, result+'0'); //해당 원소 푸쉬
                error_flag--; //스택 안의 원소 갯수
                break;
            case '%':
                a = Pop(stack)-'0';
                b = Pop(stack)-'0';
                if (a == 0) { //분모가 0일 경우 에러 출력
                    fprintf(fout, "\nError : Invalid Postfix Expression, Divide by Zero!\n");
                    DeleteStack(stack); //메모리 반환
                    exit(1); //강제종료
                }
                result = b%a; //나머지 정의
                fprintf(fout, " %d", result); //나머지값 출력
                Push(stack, result+'0'); //해당 원소 푸쉬
                error_flag--; //스택 안의 원소 갯수
                break;
            default:
                fprintf(fout, " %d", input_str[i]-'0');//숫자 나올때 출력
                Push(stack, input_str[i]); //숫자 스택에 푸쉬
                error_flag++; //스택 안의 원소 갯수
                break;
        }
        i++;
    }
    if (error_flag > 1) {  //스택 안의 원소가 존재한다면 에러 출력
        fprintf(fout, "\nInvalid Postfix Expression, %d numbers are left!", error_flag);
        DeleteStack(stack); //메모리 반환
        exit(1);  //강제종료
    }
    fprintf(fout, "\nEvaluation Result : %d\n", Top(stack)-'0');
    fclose(fin);
    fclose(fout);
    DeleteStack(stack);
}
