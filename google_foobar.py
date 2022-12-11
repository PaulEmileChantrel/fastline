

def x_y_map(x,y):
    #get coordinate i,j and return the position number
    return x+y*8

def x_y_unmap(n):
    x = n%8
    y = int((n-x)/8)
    return [x,y]



def make_tree():
    #+1,+8
    #+16,+1
    tree = {}

    for i in range(64):
        position = []
        coord = x_y_unmap(i)

        x = coord[0]
        y = coord[1]
        r12 = x+1
        if r12 <=7 and r12>=0:
            r1 = y + 2
            r2 = y -2
            if r1<=7 and r1>=0:
                position += [x_y_map(r12,r1)]
            if r2 <=7 and r2>=0:
                position += [x_y_map(r12,r2)]
        r34 = x-1
        if r34 <=7 and r34 >=0:
            r3 = y + 2
            r4 = y -2
            if r3<=7 and r3>=0:
                position += [x_y_map(r34,r3)]
            if r4 <=7 and r4>=0:
                position += [x_y_map(r34,r4)]
        r56 = x+2
        if r56 <=7 and r56 >=0:
            r5 = y + 1
            r6 = y -1
            if r5<=7 and r5 >=0:
                position += [x_y_map(r56,r5)]
            if r6 <=7 and r6 >=0:
                position += [x_y_map(r56,r6)]
        r78= x-2
        if r78 <=7 and r78 >=0:
            r7 = y + 1
            r8 = y -1
            if r7<=7 and r7 >=0:
                position += [x_y_map(r34,r7)]
            if r8 <=7 and r8 >=0:
                position += [x_y_map(r78,r8)]
        tree[str(i)] = position
    return tree




def solution(src, dest):
    #Your code here
    count = 0
    if src == dest:
        return count

    tree = make_tree()
    print(tree)

    possibilities = tree[str(src)]

    while count <20:
        count+=1
        if dest in possibilities:
            print(count)
            return count
        else:
            p = possibilities
            possibilities = []
            for i in p:
                possibilities+=tree[str(i)]

# solution(1,2)
# print(x_y_unmap(44))
# print(x_y_map(2,6))

def solution2(map):
    # Your code here

    q = [[0,0,1,False]] #[x,y,count,flag]
    visited = {}
    in_queue = {'[0,0,False]':True}
    x_max = len(map[0])-1
    y_max = len(map)-1

    while q :
        next_coor = q.pop(0)
        print(q)
        x = next_coor[0]
        y = next_coor[1]


        count = next_coor[2]
        flag = next_coor[3]

        visited[str([x,y,flag])] = True
        print(x,y,count,flag)

        if x==x_max and y==y_max:
            return count

        for [i,j] in [[x+1,y],[x,y+1],[x-1,y],[x,y-1]]:
            if 0<=i<=x_max and 0<=j<=y_max and map[j][i]==0 :
                if not visited.get(str([i,j,flag])) and not in_queue.get(str([i,j,flag])):
                    in_queue[str([i,j,flag])] = True
                    q.append([i,j,count+1,flag])

            elif 0<=i<=x_max and 0<=j<=y_max and not flag and not visited.get(str([i,j,True])) and not in_queue.get(str([i,j,flag])):
                in_queue[str([i,j,True])] = True
                q.append([i,j,count+1,True])

    return -1


map = [[0, 1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0], [1, 1, 1, 0]]
map2 = [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0], [0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]]
map3 = [[0, 1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0],[0, 1, 0, 1, 0, 1, 0],[0, 1, 0, 1, 0, 1, 0]]
#print(solution2(map))

import collections
def solution2(n):
    # Your code here

    n = int(n)
    q = [[n,0]]
    visited = {n:True}

    while q:
        next_n = q.pop(0)

        count = next_n[1]
        next_n = next_n[0]
        visited[next_n] = True

        if next_n ==1:
            return count

        for i in [next_n-1,next_n+1]:

            if 0<=i<=309 and not visited.get(i):
                q.append([i,count+1])
        if next_n%2 ==0 and not visited.get(next_n//2):
            q.append([next_n//2,count+1])

    return -1


# def solution(n):
#     # Your code here
#     n = int(n)
#     count = 0
#     while n!=1:
#         if n%2 == 0:
#             n = n//2
#         else:
#             n-=1
#         count +=1
#     return count
# for i in range(310):
#     print(solution(i))

def solution(n):
    n = int(n)
    count = 0

    while n!=1:


        if n%2 ==0:

            n = n//2

        elif n==3 or n%4==1:
            n = n+1
        else:
            n=n-1

        count+=1

    return count
# print(solution(10**300))
# print(solution2(10**300))

def solution(l):
    # Your code here
    #brute-force
    l.sort()
    lucky_list = []
    col = collections.Counter(l)

    for k in range(len(l)-1,1,-1):
        lk,lj,li = l[k],0,0
        print(k)
        for j in range(k-1,0,-1):
            print(j)
            if lk%l[j]==0:
                lj,li = l[j],0
                for i in range(j-1,-1,-1):
                    print(i)
                    if lj%l[i]==0:
                        li = l[i]
                        lucky_list.append([li,lj,lk])


    print(lucky_list)

    return len(lucky_list)

def solution(l):
    # Your code here
    #brute-force
    factoriel = [1,1,2]
    def fact(n):

        while len(factoriel)-1<n:
            factoriel.append(factoriel[-1]*(len(factoriel)))

        return factoriel[n]

    def combinatory(n,p):
        if p>n:
            return 0
        elif p==n:
            return 1

        return fact(n)/(fact(n-p)*fact(p))

    def findlucky(n,sub_list):
        if sub_list ==[]:
            return 0
        l_max = sub_list[-1]
        i = 2
        count = 0

        while n*i<=l_max:
            if col.get(n*i):
                count+=col[n*i]
            # print('f3',n,n*i,count,sub_list)

            i+=1
        return count

    def find2lucky(k,sub_list):
        if sub_list ==[]:
            return 0
        l_max = sub_list[-1]
        i = 2
        count = 0
        while k*i<=l_max:
            if col.get(k*i):
                m = col[k]
                if len(l[m:])>=2 and m>=2:
                    count += combinatory(m,2)
                if len(l[m:])>=1:

                    count += m*findlucky(k*i,sub_list[m:])
                sub_list = sub_list[m:]
                #print('f2:',k,k*i,m)

            i+=1
        return count


    l.sort()
    lucky_num = 0
    col = collections.Counter(l)

    for k,n in col.items():

        if col.get(k):

            lucky_num += combinatory(n,3)

            if len(l[n:])>=2:
                lucky_num += combinatory(n,2)*findlucky(k,l[n:])

            if len(l[n:])>=1:
                lucky_num += n*find2lucky(k,l[n:])
            print(k,n,lucky_num)

            l = l[n:]



    return int(lucky_num)

print(solution([1,2,3,4,5,6]))#3
print(solution([1,2,2,3,4,5,6]))#7
#print(solution([1,1,1]))#1
#print(solution([1,1,1,1]))#4
