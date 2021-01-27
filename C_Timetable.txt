#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int matrix[24][7] = { NULL };  //행렬 형식의 전역 정수형 배열을 선언합니다
static int ct = 0;  //전역 정수형 변수를 선언합니다(control의 의미)
static int ma = 0;  //전역 정수형 변수를 선언합니다(할당할 때 사용)
static int st_del = 0;  //전역 정수형 변수를 선언합니다(delete 함수에서 사용)
typedef struct {    //구조체를 선언합니다
	char subject[20];  //과목을 저장
	char professor[20];  //교수 성함을 저장
	int room, grade, classes;   //강의실, 학점, 수업횟수를 저장
} NAME;   //NAME이라고 지정합니다
typedef struct {   //수업횟수 저장에 영향을 받는 구조체를 선언합니다
	int whichday, classstart, howlong;  //수업날짜, 수업시작시간, 강의시간을 저장
} NAME2;    //NAME2라고 지정합니다
int compare(char* compare);   //월화수목금토일 요일을 코드화시키는 함수
void printTimeline(int day1st_code, int day7th_code); //최초 요일 입력받아 시간표 맨윗쪽 생성 함수
void printTimeline2(int time1st, int time7th, int day1st_code, int day7th_code);  //최초 비어있는 시간표 출력 함수
void printlines(int day1st_code, int day7th_code);  //요일에 따라 시간표 크기(--- 출력)를 결정하는 함수
NAME* inputSub(NAME* a, int day1st_code, int day7th_code);   //과목, 교수성함, 강의실 등 저장하는 함수
NAME2** inputSub2(NAME* a, NAME2** b, int day1st_code, int day7th_code, int time1st, int time7th); //수업날짜 등을 저장하는 함수
void printdelete(NAME* a, NAME2** b, int day1st_code, int day7th_code, int time1st, int time7th); //과목을 지우는 함수
void printTimeline3(NAME* a, NAME2** b, int day1st_code, int day7th_code, int time1st, int time7th); //최종적으로 시간표 완성하는 함수
void Current(NAME* a, NAME2** b, int day1st_code, int day7th_code, int time1st, int time7th);  //현재 시간표에 따른 공강시간 등 표시하는 함수
int main() {
	char day1st[5]; //시작요일을 입력받을 char형 변수입니다
	char day7th[5]; //마지막요일을 입력받을 char형 변수입니다
	int time1st, time7th; //시간표 시작시간, 끝나는시간 입력받을 int형 변수입니다
	while (1) {  //잘못된 값을 확인하기 위해 무한루프를 돌립니다
		printf("시간표의 시작요일과 마지막요일을 입력하세요 ");
		scanf("%s %s", day1st, day7th);  //시작요일과 마지막요일을 입력받습니다
		if ((compare(day1st) <= compare(day7th)) && compare(day1st) > 0 && compare(day7th) > 0 && compare(day1st) < 8 && compare(day7th) < 8) {
			break;  //시작요일, 마지막요일을 compare라는 생성된 함수를 통해 비교하여 시작요일이 마지막요일보다 앞에 있거나 월~일에 포함되지 않은 경우,
		}           //무한루프를 반복합니다. 조건에 부합하면 break로 탈출합니다.
		else printf("잘못 입력하였습니다. 다시 입력하세요.\n");
	}
	while (1) {   //잘못된 값을 확인하기 위해 무한루프를 돌립니다
	printf("시작시각과 종료시각을 입력하세요 ");
	scanf("%d %d", &time1st, &time7th);  //시작시간과 끝나는시간을 입력받습니다
	if (time1st < time7th && time7th < 25 && time1st>0) break; //시작시간이 마지막시간보다 크거나 두 값이 1~24시 범위를 벗어날 경우 무한루프를 돌립니다
	else printf("잘못 입력하였습니다. 다시 입력하세요.\n");
	}
	printTimeline(compare(day1st), compare(day7th));  //최초 요일 입력받아 시간표의 틀을 짜는 함수입니다
	printTimeline2(time1st, time7th, compare(day1st), compare(day7th));  //이어서 비어있는 시간표를 출력하는 함수입니다
	NAME* a = NULL;  //NAME형 포인터를 선언합니다
	NAME2** b = NULL;  //NAME2형 포인터를 선언합니다
	while (1) {     //다음 선택지에 따라 무한루프를 돌립니다
		int choice;  //선택지를 입력받을 int형 변수입니다
		printf("[시간표] 1.과목입력 2.과목삭제 3.시간표현황 0.종료 ");
		scanf("%d", &choice);  //선택지를 입력받습니다
		Current(a, b, compare(day1st), compare(day7th), time1st, time7th);
		if (choice == 1) {   //1을 입력받은 경우
			a = inputSub(a, compare(day1st), compare(day7th));  //과목, 교수성함, 강의실 등을 입력받는 함수 호출
			b = inputSub2(a, b, compare(day1st), compare(day7th), time1st, time7th); //수업날짜 등을 입력받는 함수 호출
		}
		else if (choice == 2) printdelete(a, b, compare(day1st), compare(day7th), time1st, time7th); //2를 입력받은경우 과목을 지우는 함수 호출
		else if (choice == 3) printTimeline3(a, b, compare(day1st), compare(day7th), time1st, time7th); //3을 입력받은경우 현 시간표 출력하는 함수 호출
		else if (choice == 0) {  //0을 입력받은경우
			break;     //무한루프를 탈출하여 프로그램을 마칩니다
		}
	}
	for (int i = 0; i < ct; i++) {
		free(b[i]);  //inputSub2함수에서 할당했던 메모리를 free
	}
	free(a);	//inputSub함수에서 할당했던 메모리를 free
	free(b);    //inputSub2함수에서 할당했던 메모리를 free
	return 0;
}
int compare(char* compare) {  //요일을 코드화하는 함수입니다
	int result_comp = 0;      //int형 변수를 0이라 둡니다
	if (strcmp(compare, "월") == 0) result_comp = 1; //월을 숫자 1로 둡니다
	else if (strcmp(compare, "화") == 0) result_comp = 2; //화를 숫자 2로 둡니다
	else if (strcmp(compare, "수") == 0) result_comp = 3; //수를 숫자 3로 둡니다
	else if (strcmp(compare, "목") == 0) result_comp = 4; //목을 숫자 4로 둡니다
	else if (strcmp(compare, "금") == 0) result_comp = 5; //금을 숫자 5로 둡니다
	else if (strcmp(compare, "토") == 0) result_comp = 6; //토를 숫자 6로 둡니다
	else if (strcmp(compare, "일") == 0) result_comp = 7; //일을 숫자 7로 둡니다
	return result_comp;  //변환한 값을 리턴합니다
}
void printTimeline(int day1st_code, int day7th_code) {  //최초 시간표 틀을 짜는 함수
	printlines(day1st_code, day7th_code);  //요일에 따라 줄(---)을 치는 함수 호출
	printf("\n");
	for (int i = day1st_code; i < day7th_code + 1; i++) {  //코드화한 요일에 따라 다음을 실행합니다
		char day_samp[5];  //시간표에 출력하기 위해 char형 변수를 선언합니다
		if (i == 1) strcpy(day_samp, "월");  //코드가 1인 경우 월을 대입합니다
		else if (i == 2) strcpy(day_samp, "화"); //코드가 2인 경우 화를 대입합니다
		else if (i == 3) strcpy(day_samp, "수"); //코드가 3인 경우 수를 대입합니다
		else if (i == 4) strcpy(day_samp, "목"); //코드가 4인 경우 목을 대입합니다
		else if (i == 5) strcpy(day_samp, "금"); //코드가 5인 경우 금을 대입합니다
		else if (i == 6) strcpy(day_samp, "토"); //코드가 6인 경우 토를 대입합니다
		else if (i == 7) strcpy(day_samp, "일"); //코드가 7인 경우 일을 대입합니다
		printf("      %s", day_samp);  //위 규칙에 따라 출력합니다
	}
	printf("\n");
	printlines(day1st_code, day7th_code);  //요일에 따라 줄을 치는 함수
	printf("\n");
}
void printTimeline2(int time1st, int time7th, int day1st_code, int day7th_code) {  //최초 시간표 틀을 짜는 함수
	for (int i = time1st; i < time7th + 1; i++) { //코드화한 요일에 따라 다음을 실행합니다
		if (i < 10) {  //시작시간이 10시보다 이를경우
			printf("0%d\n", i); //x시면 0x시로 표시하도록 합니다
			printlines(day1st_code, day7th_code);  //요일에 따라 줄을 치는 함수
			printf("\n");
		}
		else {   //시작시간이 10시거나 그보다 늦을 경우
			printf("%d\n", i); //그대로 표시하도록 합니다
			printlines(day1st_code, day7th_code);  //요일에 따라 줄을 치는 함수
			printf("\n");
		}
	}
}
NAME* inputSub(NAME* a, int day1st_code, int day7th_code) {   //과목 등을 입력받아 구조체에 저장하는 함수
	ct++;   //전역변수를 1 증가시킵니다
	a = (NAME*)realloc(a, ct * sizeof(NAME)); //a를 실행할때마다 재할당하여 크기를 늘려줍니다
	printf("과목명 : ");
	scanf("%s", a[ct - 1].subject);  //과목 이름을 입력받습니다
	printf("교수명 : ");
	scanf("%s", a[ct - 1].professor); //교수 성함을 입력받습니다
	printf("강의실 : ");
	scanf("%d", &a[ct - 1].room); //강의실을 입력받습니다
	printf("학점 : ");
	scanf("%d", &a[ct - 1].grade); //학점을 입력받습니다
	while (1) {  //무한루프를 돌립니다
		printf("주당강의횟수 : ");
		scanf("%d", &a[ct - 1].classes); //강의횟수를 입력받습니다
		if (a[ct-1].classes < day7th_code-day1st_code+2) break;  //주당 강의수가 요일 수를 초과하면 무한루프를 반복합니다
	}
	return a;  //a를 반환합니다
}
NAME2** inputSub2(NAME* a, NAME2** b, int day1st_code, int day7th_code, int time1st, int time7th) { //수업 날짜 등을 구조체에 저장하는 함수
	b = (NAME2**)realloc(b, (a[ct - 1].classes) * sizeof(NAME2));  //b를 재할당하여 크기를 늘려줍니다
	b[ct - 1] = (NAME*)malloc((ct) * sizeof(NAME)); //행에 대해서도 재할당을 해줍니다
	char stp[5];  //강의요일을 입력받을 변수를 선언합니다
	for (int i = 0; i < a[ct - 1].classes; i++) {  //위에 입력받은 강의횟수만큼 실행합니다
		printf("<강의%d>\n", i + 1);
		while (1) {   //잘못 입력한것 처리하기 위해 무한루프를 돌립니다
			printf("강의요일 : ");
			scanf("%s", stp);   //강의요일을 입력받습니다
			b[ct - 1][i].whichday = compare(stp);  //요일을 코드화하는 함수로 보내 정수로 구조체에 넣습니다
			if (compare(stp) < day1st_code || compare(stp) > day7th_code) {
				printf("잘못 입력\n");    //요일 범위를 벗어나는 경우 오류문을 출력하고
			} //무한루프를 계속 돌립니다
			else break;   //조건에 맞으면 무한루프를 탈출합니다
		}
		while (1) {
			printf("시작시간 : ");
			scanf("%d", &b[ct - 1][i].classstart);  //구조체에 시작시간을 입력받습니다
			if (b[ct - 1][i].classstart < time1st && b[ct - 1][i].classstart > time7th) {
				printf("잘못 입력\n");    //시간 범위를 벗어나는 경우 오류문을 출력하고
			} //무한루프를 계속 돌립니다
			else break;   //조건에 맞으면 무한루프를 탈출합니다
		}
		while (1) {
			printf("강의시간 : ");
			scanf("%d", &b[ct - 1][i].howlong);   //구조체에 강의시간을 입력받습니다
			if (b[ct-1][i].howlong>1+time7th-b[ct-1][i].classstart) {
				printf("잘못 입력\n");    //시간 범위를 벗어나는 경우 오류문을 출력하고
			} //무한루프를 계속 돌립니다
			else break;   //조건에 맞으면 무한루프를 탈출합니다
		}
	}
	int case1 = 1;   //정수형 변수를 1이라 둡니다
	for (int z = 0; z < a[ct - 1].classes; z++) {  //위에 입력받은 강의횟수만큼 실행
		for (int i = 0; i < time7th - time1st + 1; i++) {    //행렬 형태의 시간표의 행만큼 실행
			for (int j = 0; j < day7th_code - day1st_code + 1; j++) {    //행렬 형태의 시간표의 열만큼 실행
				if (i >= b[ct - 1][z].classstart - time1st && i < b[ct - 1][z].classstart - time1st + b[ct - 1][z].howlong && j == b[ct - 1][z].whichday - day1st_code) {
					// 행렬 상으로 강의시간 등을 고려하여 숫자로 표현한 시간표를 고려합니다
					if (matrix[i][j] != 0) case1 = 0;  //행렬에 0이 아닌 성분 존재시, case1을 0이라 둡니다
				}
			}
		}
	}
	if (case1 == 0) {
		printf("시간표 겹침!!!\n");  //행렬에 0이 아닌 성분 존재시 오류문 출력후 아래 문장을 건너뜁니다
		a[ct-1].grade = 0;     //학점을 0으로 바꿉니다
		for (int i = 0; i < a[ct-1].classes; i++) b[ct-1][i].howlong = 0;   //강의 길이도 0으로 초기화합니다
		st_del++; //전역변수에 1을 더합니다
	}
	else if (case1 == 1) {   //case1이 그대로 1이면
		for (int z = 0; z < a[ct - 1].classes; z++) {  //위에 입력받은 강의횟수만큼 실행
			for (int i = 0; i < time7th - time1st + 1; i++) {   //행렬 형태의 시간표의 행 크기만큼 실행
				for (int j = 0; j < day7th_code - day1st_code + 1; j++) {    //행렬 형태의 시간표의 열 크기만큼 실행
					if (i >= b[ct - 1][z].classstart - time1st && i < b[ct - 1][z].classstart - time1st + b[ct - 1][z].howlong && j == b[ct - 1][z].whichday - day1st_code) {
						// 행렬 상으로 강의시간 등을 고려하여 숫자로 표현한 시간표를 고려합니다
						matrix[i][j] = ct;   //행렬에서 조건에 맞는 성분들을 전역변수 ct로 채웁니다
					}
				}
			}
		}
		printTimeline3(a, b, day1st_code, day7th_code, time1st, time7th);  //이후 최종적으로 시간표를 생성하는 함수를 호출합니다
	}
	return b;   //b를 리턴합니다
}
void printdelete(NAME* a, NAME2** b, int day1st_code, int day7th_code, int time1st, int time7th) {  //과목을 지우는 함수
	char del[20], del_str[5] = { NULL };    //지울 과목명, 요일을 받을 char형 변수 2개 선언
	int del1 = -1, del2 = 0, sum = 0, finalch;  //int형 변수 4개 선언
	printf("과목명 : ");
	scanf("%s", del);    //과목명을 입력받습니다
	for (int k = 0; k < ct+1; k++) {    //과목을 입력받은 횟수만큼 실행
		if (strcmp(del, a[k].subject) == 0) del1 = k;    //입력받은 과목이 존재한다면 그때 위치를 del1에 저장합니다
	}
	for (int z = 0; z < a[del1].classes; z++) {   //강의 횟수만큼 실행
		for (int i = 0; i < time7th - time1st + 1; i++) {   //시간표의 행 크기만큼 실행
			for (int j = 0; j < day7th_code - day1st_code + 1; j++) {    //시간표의 열 크기만큼 실행
				if (i >= b[del1][z].classstart - time1st && i < b[del1][z].classstart - time1st + b[del1][z].howlong && j == b[del1][z].whichday - day1st_code) {
					if (matrix[i][j] == 0) del2 = 1;   //이미 지워진 경우 행렬의 성분은 0으로 둘것이고, 이 경우 del2를 1로 둡니다
				}
			}
		}
	}
	if (del2 == 1 || del1 == -1) { //이미 지워진 과목을 입력하였거나 없는 과목 입력시
		printf("존재하지 않는 과목입니다.\n");  //오류를 출력합니다
		return;
	}
	else if (del1 > -1) {   //del1값이 바뀐 경우
		printf("다음과목을 삭제하시겠습니까?\n");
		printf("교수명 : %s\n", a[del1].professor);    //교수 성함을 입력받습니다
		printf("강의실 : %d\n", a[del1].room);    //강의실을 입력받습니다
		printf("학점 : %d\n", a[del1].grade);    //학점을 입력받습니다
		for (int i = 0; i < a[del1].classes; i++) sum += b[del1][i].howlong;    //강의 길이를 입력받습니다
		printf("총 강의시간 : %d ( ", sum);
		for (int i = 0; i < a[del1].classes; i++) {   //강의 횟수만큼 코드화한 요일을 실제 요일로 바꿉니다
			if (b[del1][i].whichday == 1) strcpy(del_str, "월");  //1은 월요일
			else if (b[del1][i].whichday == 2) strcpy(del_str, "화");  //2은 화요일
			else if (b[del1][i].whichday == 3) strcpy(del_str, "수");  //3은 수요일
			else if (b[del1][i].whichday == 4) strcpy(del_str, "목");  //4은 목요일
			else if (b[del1][i].whichday == 5) strcpy(del_str, "금");  //5은 금요일
			else if (b[del1][i].whichday == 6) strcpy(del_str, "토");  //6은 토요일
			else if (b[del1][i].whichday == 7) strcpy(del_str, "일");  //7은 일요일
			printf("%s", del_str);    //요일 출력
			printf(" %d ", b[del1][i].classstart);   //시작시간 출력
		}
		printf(")\n(삭제-1 / 취소-2) ");
		scanf("%d", &finalch);   //최종 선택지를 입력받습니다
		if (finalch == 1) {    //삭제를 선택한 경우
			for (int z = 0; z < a[del1].classes; z++) {     //강의 횟수만큼 실행
				for (int i = 0; i < time7th - time1st + 1; i++) {   //시간표의 행 크기만큼 실행
					for (int j = 0; j < day7th_code - day1st_code + 1; j++) {    //시간표의 열 크기만큼 실행
						if (i >= b[del1][z].classstart - time1st && i < b[del1][z].classstart - time1st + b[del1][z].howlong && j == b[del1][z].whichday - day1st_code) {
							matrix[i][j] = 0;   //지울 과목에 대응되는 행렬의 성분을 0으로 둡니다
						}
					}
				}
			}
			a[del1].grade = 0;     //학점을 0으로 바꿉니다
			for (int i = 0; i < a[del1].classes; i++) b[del1][i].howlong = 0;   //강의 길이도 0으로 초기화합니다
			st_del++;    //전역변수를 1 증가시킵니다 = 1과목 삭제되었습니다
			printf("삭제완료\n");
		}
		else if (finalch == 2) return;   //선택지에서 취소를 선택하면 지우지 않고 돌아갑니다
	}
}
void printTimeline3(NAME* a, NAME2** b, int day1st_code, int day7th_code, int time1st, int time7th) {
	if (ct==0) {   //과목을 입력받지 않은 경우
		printf("시간표가 비었습니다!\n");   //오류를 출력합니다
		return 0;
	}
	printTimeline(day1st_code, day7th_code);   //최초 요일을 입력받아 시간표의 상단 구성하는 함수
	for (int k = 1; k < ct + 1; k++) {     //구조체에 입력받은 횟수만큼 실행
		(a[ct - 1].subject)[4] = NULL;     //우선 4바이트만 저장합니다
		if (a[ct - 1].subject[0] >= 'A' || a[ct - 1].subject[2] >= 'A') (a[ct - 1].subject)[3] = NULL; //양 끝의 문자가 영어인 경우 3바이트만 저장합니다
	}
	for (int i = 0; i < time7th - time1st + 1; i++) {    //시간표의 행에 관해 실행
		if (i + time1st < 10) printf("0%d   ", i + time1st);    //시작시간이 10시보다 이를경우 x시면 0x시로 표시하도록 합니다
		else printf("%d   ", i + time1st);     //시작시간이 10시거나 그보다 늦을 경우 그대로 표시합니다
		for (int j = 0; j < day7th_code - day1st_code + 1; j++) {   //시간표의 열에 관해 실행
			int stk = 0, stk2 = 0;    //int형 변수 2개 선언
			for (int k = 1; k < ct + 1; k++) {   //구조체에 입력받은 횟수만큼 실행
				if (matrix[i][j] == k) {    //행렬에 해당되는 
					printf("%s", a[k - 1].subject);  //행렬 요소에 대응되는 과목 출력
					stk2 = 1;   //stk2를 1로 둡니다
				}
				stk = strlen(a[k - 1].subject);  //과목의 크기를 구해 stk에 집어넣습니다
			}
			if (stk == 4 && stk2 == 0) printf("    ");  //과목의 길이, stk2의 값에 따라 크기가 다른 공백 출력
			else if (stk == 3 && stk2 == 0) printf("   ");  //과목의 길이, stk2의 값에 따라 크기가 다른 공백 출력
			if (stk == 4) printf("    ");   //과목의 길이의 값에 따라 크기가 다른 공백 출력
			else if (stk == 3) printf("     ");  //과목의 길이의 값에 따라 크기가 다른 공백 출력
		}
		printf("\n");
		printlines(day1st_code, day7th_code);  //줄 치는 함수 호출
		printf("\n");
	}
}
void printlines(int day1st_code, int day7th_code) {   //요일에 따라 줄(---)을 치는 함수
	if (day1st_code==day7th_code) printf("---------");   //요일이 하루일 경우
	else if (day7th_code-day1st_code==1) printf("------------------");   //요일이 2일인 경우
	else if (day7th_code-day1st_code==2) printf("---------------------------");   //요일이 3일인 경우
	else if (day7th_code-day1st_code==3) printf("------------------------------------");   //요일이 4일인 경우
	else if (day7th_code-day1st_code==4) printf("---------------------------------------------");   //요일이 5일인 경우
	else if (day7th_code-day1st_code==5) printf("------------------------------------------------------");   //요일이 6일인 경우
	else if (day7th_code-day1st_code==6) printf("---------------------------------------------------------------");   //요일이 7일인 경우
}
void Current(NAME* a, NAME2** b, int day1st_code, int day7th_code, int time1st, int time7th) {  //현재 시간표에 따른 공강시간 등 표시하는 함수
	if (ct==0) return 0;   //최초에는 아무것도 실행하지 않습니다  
	int sum_grade=0, sum_howl=0;   //int형 변수 2개를 선언
	printf("총 과목 : %d과목\n", ct-st_del);    //전역변수 2개를 통해 입력받은 횟수-삭제한 횟수를 통해 과목수를 확인
	for (int i = 0; i < ct; i++) sum_grade += a[i].grade;    //입력받은 학점수를 모두 더합니다
	printf("총 학점 : %d학점\n", sum_grade);   //이를 출력합니다
	for (int i=0; i<ct; i++) {     //입력받은 횟수에 한해
	 	for (int j=0; j<a[i].classes; j++) {   //강의 횟수에 한해
			sum_howl+=b[i][j].howlong;    //수업이 진행되는 시간을 모두 더해줍니다
		}
	}
	printf("총 강의시간 : %d시간\n", sum_howl);  //수업 진행시간을 모두 합한 것을 출력
	printf("공강시간 : %d시간\n", (day7th_code-day1st_code+1)*(time7th-time1st+1)-sum_howl);  //시간표상 총 시간-수업 진행시간의 합 통해 공강시간확인
}
