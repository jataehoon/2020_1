import collections
def solution(participant, completion):
    answer = collections.Counter(participant) - collections.Counter(completion)
    return list(answer.keys())[0]
'''
def solution(participant, completion):
    for i in range(0, len(participant)):
        for j in range(0, len(participant)-1):
            if participant[i] == completion[j]:
                participant[i]=''
                completion[j]=''
    print(participant)
    for k in range(0, len(participant)):
        if participant[k]!='':
            answer = participant[k]
    return answer
'''