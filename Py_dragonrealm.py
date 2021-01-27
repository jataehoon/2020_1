import time
import random

def displayintro():
    print('You are in a land full of dragons, In front of you,')
    print('you see two caves. In one cave, the dragon is friendly')
    print('and will share his treasure with you. The other dragon')
    print('is greedy and hungry, and will eat you on sight.')
    print()

def choosecave():
    cave = ''
    while cave != '1' and cave != '2':
        print('Which cave will you go into? (1 or 2)')
        cave=input()
    return cave

def checkcave(chosencave):
    print('You approach the cave...')
    time.sleep(1)
    print('It is dark and spooky...')
    time.sleep(1)
    print('A large drago jumps out in front of you! He opens his jaws and...')
    print()
    time.sleep(1)
    friendlycave=random.randint(1,2)
    if chosencave == str(friendlycave):
        print('Gives you his treasure!')
    else:
        print('Gobbles you down in one bite!')

playagain = 'yes'
while playagain == 'yes' or playagain == 'y':
    displayintro()
    cavenumber=choosecave()
    checkcave(cavenumber)
    print('Do you want to play again?(yes or no)')
    playagain=input()
