#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
char cipher(char* pwd, int spin);   //cipher라는 함수를 알려줍니다
char decipher(char* pwd, int spin); //decipher라는 함수를 알려줍니다
int main()
{
    char pw[100];  //char형에 크기 100짜리 변수 pw를 선언합니다
    char *pwd=pw;  //char형 포인터 pwd를 선언합니다
    int spin, choice;   //이동횟수, 선택지를 위한 int형 변수를 선언합니다
    printf("숨기고자 하는 메세지 : "); //메세지 입력에 관한 안내문입니다
    scanf("%[^\n]s",pw);  //스페이스바를 포함한 열을 저장하여 pw에 저장합니다
    while (1){    //무한 반복문을 정의합니다
        printf("\n선택하세요 0.종료 1.암호화 2.복호화 : ");  //선택지를 띄웁니다
        scanf("%d",&choice);  //숫자를 choice에 입력받습니다
        if (choice == 0) break;  //choice가 0인 경우 반복문을 탈출합니다
        if (choice == 1){   //choice가 1인 경우
        printf("이동횟수(양/음수 가능) : "); //이동횟수 입력을 위한 안내문입니다
        scanf("%d",&spin);  //이동횟수를 입력받아 spin에 저장합니다
        cipher(pw, spin);   //cipher라는 함수를 호출합니다
        printf("%s",pw);    //함수를 통해 암호화하여 출력합니다
        }
        else if (choice == 2){  //choice가 2인 경우
        printf("이동횟수(양/음수 가능) : "); //이동횟수 입력을 위한 안내문입니다
        scanf("%d",&spin);  //이동횟수를 입력받아 spin에 저장합니다
        decipher(pw, spin); //decipher라는 함수를 호출합니다
        printf("%s",pw);    //함수를 통해 해독하여 출력합니다
        }
    }
    return 0;  //프로그램을 마칩니다
}
char cipher(char* pwd, int spin){  //cipher라는 암호화를 위한 함수를 정의합니다
    int count=0;  //int형 변수 count를 선언합니다
    while (pwd[count] != 0) count++; //다음 식을 통해 메시지의 길이를 구합니다
    for (int i=0;i<count;i++) {  //메시지의 길이만큼 아래를 실행합니다
        if (pwd[i]>96 && pwd[i]<123) {  //아스키 코드상 메시지가 소문자인 경우
            if (pwd[i]+spin>122) pwd[i]=97+(pwd[i]+spin-97)%26;/*각 문자의 아스키코드
와 spin의 합이 아스키코드로 표현되는 범위를 초과한 경우에 대한 식입니다*/
            else if (pwd[i]+spin>96 && pwd[i]+spin<123) pwd[i]=pwd[i]+spin;/*각 문자의
아스키코드와 spin의 합이 아스키코드로 표현되는 범위 안에 있는 경우에 대한 식입니다*/
            else if (pwd[i]+spin<97) pwd[i]=122+(pwd[i]+spin-96)%26;/*각 문자의
아스키코드와 spin의 합이 아스키코드로 표현되는 범위 미만인 경우에 대한 식입니다*/
        }        
        else if (pwd[i]>64 && pwd[i]<91) {  //아스키 코드상 메시지가 대문자인 경우
            if (pwd[i]+spin>90) pwd[i]=65+(pwd[i]+spin-65)%26;/*각 문자의 아스키코드
와 spin의 합이 아스키코드로 표현되는 범위를 초과한 경우에 대한 식입니다*/
            else if (pwd[i]+spin>64 && pwd[i]+spin<91) pwd[i]=pwd[i]+spin;/*각 문자의
아스키코드와 spin의 합이 아스키코드로 표현되는 범위 안에 있는 경우에 대한 식입니다*/
            else if (pwd[i]+spin<65) pwd[i]=90+(pwd[i]+spin-64)%26;/*각 문자의
아스키코드와 spin의 합이 아스키코드로 표현되는 범위 미만인 경우에 대한 식입니다*/
        }
    }
    return pwd;  //pwd를 반환합니다
}
char decipher(char* pwd, int spin){
    int count=0;  //int형 변수 count를 선언합니다
    while (pwd[count] != 0) count++; //다음 식을 통해 메시지의 길이를 구합니다
    for (int i=0;i<count;i++) {  //메시지의 길이만큼 아래를 실행합니다
        if (pwd[i]>96 && pwd[i]<123) {  //아스키 코드상 메시지가 소문자인 경우
            if (pwd[i]-spin>122) pwd[i]=97+(pwd[i]-spin-97)%26;/*각 문자의 아스키코드
와 spin의 합이 아스키코드로 표현되는 범위를 초과한 경우에 대한 식입니다*/
            else if (pwd[i]-spin>96 && pwd[i]-spin<123) pwd[i]=pwd[i]-spin;/*각 문자의
아스키코드와 spin의 합이 아스키코드로 표현되는 범위 안에 있는 경우에 대한 식입니다*/
            else if (pwd[i]-spin<97) pwd[i]=122+(pwd[i]-spin-96)%26;/*각 문자의
아스키코드와 spin의 합이 아스키코드로 표현되는 범위 미만인 경우에 대한 식입니다*/
        }        
        else if (pwd[i]>64 && pwd[i]<91) {  //아스키 코드상 메시지가 대문자인 경우
            if (pwd[i]-spin>90) pwd[i]=65+(pwd[i]-spin-65)%26;/*각 문자의 아스키코드
와 spin의 합이 아스키코드로 표현되는 범위를 초과한 경우에 대한 식입니다*/
            else if (pwd[i]-spin>64 && pwd[i]-spin<91) pwd[i]=pwd[i]-spin;/*각 문자의
아스키코드와 spin의 합이 아스키코드로 표현되는 범위 안에 있는 경우에 대한 식입니다*/
            else if (pwd[i]-spin<65) pwd[i]=90+(pwd[i]-spin-64)%26;/*각 문자의
아스키코드와 spin의 합이 아스키코드로 표현되는 범위 미만인 경우에 대한 식입니다*/
        }
    }
    return pwd;  //pwd를 반환합니다
}
