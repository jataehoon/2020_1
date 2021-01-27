#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
void login(char *id, char *pw);   //login이란 함수를 알려줍니다
char real_id[7] = "qwerty", real_pw[5] = "1234";  //실제 아이디, 비밀번호는 각각 qwerty, 1234로 정의하여 char형 전역 변수로 관리합니다
void main() {
	char id[100], pw[100];  //아이디, 비밀번호를 입력받을 char형 변수 두 개를 선언합니다
	for (int i = 0; i < 3; i++) {  //최대 3번 반복할 예정입니다
		printf("로그인하세요\n");  //로그인 안내문입니다
		printf("아이디 : ");       //아이디 입력 안내문입니다
		scanf("%s", id);    //id에 입력한 아이디를 저장합니다
		printf("암호 : ");  //암호 입력 안내문입니다
		scanf("%s", pw);    //pw에 입력한 비밀번호를 저장합니다
		login(id, pw);      //login 함수를 호출합니다
		if (strcmp(real_id, id) == 0 && strcmp(real_pw, pw) == 0) {  //실제 아이디와 입력 아이디 및 비밀번호 비교시 같을 경우
			break;    //반복문을 종료하고 같지 않으면 반복을 계속 실행합니다
		}
	}
	return 0;  //프로그램을 종료합니다
}
void login(char *id, char *pw) {
	static int try = 3;   //로그인 실패 횟수를 의미하는 정적 지역 변수를 지정하여 3으로 초기화합니다
	extern char real_id[7], real_pw[5];   //실제 아이디를 갖고있는 전역 변수를 사용합니다
	if (strcmp(real_id, id) == 0 && strcmp(real_pw, pw) == 0) {    //실제 아이디와 입력 아이디 및 비밀번호 비교시 같을 경우
		printf("\n로그인 성공!\n");    //로그인 성공을 출력합니다
	}
	else {   //실제 아이디와 입력 아이디 및 비밀번호 비교시 같지 않을 경우
		try--;    //입력 실패마다 시도횟수를 1씩 감소시킵니다
		printf("\n로그인에 실패했습니다.\n");   //실패하였다고 알려줍니다
		printf("로그인 시도 %d회 남았습니다.\n\n", try);   //남은 시도 횟수를 알려줍니다
	}
}

