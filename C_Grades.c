#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void main() {
	int prev_pt;
	double prev_gr;
	int trial;
	double sum_a = 0.000;
	int sum_b = 0;
	char pre_point;
	int point;
	int grade;
	printf("★★★単位計算機★★★\n");
	printf("今まで履修した単位(今学期除外)(ex. 80) : ");
	scanf("%d", &prev_pt);
	printf("今まで平均点数(今学期除外)(ex. 3.5) : ");
	scanf("%lf", &prev_gr);
	printf("今学期除受けた科目科目数はいくつか? ");
	scanf("%d", &trial);
	for (int i = 0; i < trial; i++) {
		while (1) {
			printf("%d番目科目点数? (小文字アルファベットで入力してください。) ", i + 1);
			scanf(" %c", &pre_point);
			if (pre_point == 'a' || pre_point == 'b' || pre_point == 'c' ||
				pre_point == 'd' || pre_point == 'e' || pre_point == 'f') break;
			else printf("入力を間違えました\n");
		}
		if (pre_point == 'a') point = 4.000;
		else if (pre_point == 'b') point = 3.000;
		else if (pre_point == 'c') point = 2.000;
		else if (pre_point == 'd') point = 1.000;
		else if (pre_point == 'f') point = 0.000;
		printf("%d番目科目単位? (3単位なら 3を入力すればいい) ", i + 1);
		scanf("%d", &grade);
		sum_a += point * grade;
		sum_b += grade;
	}
	printf("今学期履修単位 : %d\n", sum_b);
	printf("今学期の平均評点 : %.3lf\n", sum_a/sum_b);
	printf("今学期を含めた全履修単位 : %d\n", prev_pt+sum_b);
	printf("今学期を含めた全体の平均評点 : %.3lf\n", (prev_pt*prev_gr*1.000+sum_a)/(prev_pt+sum_b));
}
