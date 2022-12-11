def parityCheck(x : int) -> bool:
    num_bit = 0
    while x:
        num_bit += x&1
        x>>=1
    return num_bit%2 == 0


#
# print(parityCheck(2))
# print(parityCheck(3))
# print(parityCheck(4))
# print(parityCheck(5))

def power(x : float, y : int)-> float:
    if y ==0:
        return 1
    if y % 2 == 0:
        r = power(x,y/2)
        return r * r
    else:
        return x * power(x,y-1)


# print(power(5,0))
# print(power(5,1))
# print(power(5,2))
# print(power(5,3))
# print(power(6,4))

def partition(arr:list,pivot_index:int)->list:
    if pivot_index >=len(arr):
        return arr

    pivot = arr[pivot_index]
    arr[pivot_index],arr[0]=arr[0],arr[pivot_index]
    pivot_index = 0
    pp1 = 1
    for i in range(1,len(arr)):
        if arr[i]<=pivot:
            arr[pivot_index] = arr[i]
            arr[i] = arr[pp1]
            arr[pp1] = pivot
            pp1+=1
            if arr[pivot_index]!=pivot:
                pivot_index+=1
        print(arr,pivot_index)

    return arr


#print(partition([0,3,1,2,0,2,3,2,1,3],3))


def bestProfit(arr:list)->int:
    maxPrice = arr[-1]
    bestPrice = 0
    for i in range(len(arr)-2,0,-1):
        if arr[i]>maxPrice:
            maxPrice = arr[i]
        if maxPrice-arr[i]>bestPrice:
            bestPrice = maxPrice-arr[i]
    return bestPrice

#print(bestProfit([310,315,275,295,260,270,290,230,255,250]))
import random
def dataSample(arr:list,size:int)->list:
    if len(arr)<=size:
        return arr
    idx = 0
    for i in range(size):
        r = random.randint(idx,len(arr)-1)
        print(r)
        arr[r],arr[idx] = arr[idx],arr[r]
        idx+=1
    return arr[:idx]

#print(dataSample([310,315,275,295,260,270,290,230,255,250],3))

def spiral(m:list)-> list:

    cs = 0
    ce = len(m)-1
    rs = 0
    re = len(m[0])-1

    arr = [0]*(ce+1)*(re+1)
    len_arr = len(arr)


    l = 0
    while cs<ce:
        for i in range(cs,ce):
            arr[l] = m[rs][i]
            l+=1

        if l==len_arr:
            break

        for j in range(rs,re):
            arr[l] = m[j][ce]
            l+=1


        for i in range(ce,cs,-1):
            arr[l] = m[re][i]
            l+=1

        if l==len_arr:
            break

        for j in range(re,rs,-1):
            arr[l] = m[j][cs]
            l+=1
        rs+=1
        cs+=1
        ce-=1
        re-=1

    return arr
# print(spiral([[1,2],[3,4]]))
# print(spiral([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]))


def addOne(arr:list)->list:
    flag = False

    len_arr = -1
    while arr[len_arr]+1>=10:
        arr[len_arr] = 0
        len_arr-=1

        if len(arr)+len_arr<0:
            arr.insert(0,1)
            return arr

    arr[len_arr] = arr[len_arr]+1

    return arr
#
# print(addOne([1,2,3]))
# print(addOne([1,1,9]))
# print(addOne([1,1,9,9]))
# print(addOne([9,9,9]))

def str2int(s:str)->int:
    d = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
    n = 0
    count = 0
    for i in range(len(s)):
        if s[~i] =='-':
            n *= -1
        elif s[~i] !='+':
            digit = d[s[~i]]
            n += 10**count*digit
            count+=1
    return n

# print(str2int('-123'))
#
# print(str2int('123'))
#
# print(str2int('1230'))


def int2str(n:int)->str:
    s = ''
    arr = ['0','1','2','3','4','5','6','7','8','9']
    flag =  n <0
    n = abs(n)
    count = 0
    while n :
        d = arr[n%10]
        s = d+s
        n = n//10

    if flag :
        s = '-' + s
    else:
        s = '+' + s

    return s

# print(int2str(-1234))
# print(int2str(123))
import math

def base2base(s:str,b1:int,b2:int)->str:
    l = len(s)-1
    n = 0
    d = {'A':10,'B':11,'C':12,'D':13,'E':14,'F':15}
    for i in range(len(s)):
        if d.get(s[i]):
            a = s[i]

        else:
            n += int(s[i])*(b1**l)
        l -= 1

    new_str_end = str(n%b2)
    n -= n%b2

    power_max = 1
    while n>b2**power_max:
        power_max+=1

    new_str =''
    power_max-=1
    arr =['A','B','C','D','E','F']
    while power_max:
        x = math.floor(n/(b2**power_max))
        if x>=10:
            new_str += arr[x-10]
        else:
            new_str += str(x)
        n -= x*b2**power_max
        power_max -=1
    new_str = new_str+new_str_end
    return new_str

#print(base2base('615',7,13))

def removeAndDelete(arr:list,n:int)->list:
    na = n
    i = 0
    while i <len(arr):
        if arr[i]=='b':
            arr[i:-1] = arr[i+1:]
            arr[-1] = ''
            na -= 1
        else:
            i+=1
    nb = n-na
    if na<nb:
        len_arr = len(arr)+na-nb
    else:
        len_arr = len(arr)

    last_letter = len_arr - na -1
    i = len_arr-1
    while i != last_letter:
        if arr[last_letter] =='a':
            arr[i] = 'd'
            arr[i-1] = 'd'
            i-=2
            last_letter-=1
        else:
            arr[i] = arr[last_letter]
            i-=1
            last_letter-=1
    return arr

# print(removeAndDelete(['a','a','b','b','c','c'],4))
# print(removeAndDelete(['a','b','b','c','c'],3))
# print(removeAndDelete(['c','f','a','b','b','c','c'],3))
# print(removeAndDelete(['a','a','b','c','c',''],3))


class ListNode:
    def __init__(self, data = 0,next = None):
        self.data = data
        self.next = next

def search_list(L : ListNode, key:int)->None:
    while L and L.data != key:
        L = L.next

    return L
def insert_node(L : ListNode, N : ListNode)->None:
    #after L
    N.next = L.next
    L.next = N
    #or before L
    N.next = L
def delete_after(L:ListNode)->None:
    L.next = L.next.next


def merge_linked_list(L1:ListNode,l2:ListNode)->None:

    #Comparaison of the 2 firsts node
    # if L2.data<L1.data:
    #     L = L2.next
    #     L2.next = L1
    #     L1 = L2
    #     L2 = L
    #     L1 = L1.next
    #
    # while L1 != None :
    #     if L2.data<L1.next.data and L1.data<L2.data:
    #         L = L2.next
    #         L2.next = L1.next
    #         L1.next = L2
    #         L2 = L
    #
    #     L1 = L1.next
    #
    #     if L2 == None:
    #         return L1
    #
    # L1.next = L2
    # return L1

    L = ListNode()
    head = L
    while L1 and L2 :
        if L1.data>=L2.data:
            L.next,L2 = L2,L2.next

        else :
            L.next,L1 = L1,L1.next

    if L2 == None:
        L.next = L1
    else:
        L.next = L2

    return head.next


def reversal(L:ListNode,s:int,f:int)->ListNode:
    if f<s:
        return L


    head = sub_link_head = L


    for i in range(1,s):
        sub_link_head = sub_link_head.next

    sub_tail = sub_link_head.next
    sub_L = sub_link_head.next

    for _ in range(f-s+1):
        sub_tail = sub_tail.next

    for _ in range(f-s+1):
        L1,sub_L.next = sub_L.next,sub_tail
        sub_tail,sub_L = sub_L,L1

    sub_link_head.next = sub_tail

    return head


def findCycle(L:ListNode)->None:

    s = f = L
    s = s.next
    f = f.next.next

    while f!=s and f.next:
        s = s.next
        f = f.next.next

        if f.next.next == None:
            return None
    if s == f:
        f = L
        while f!=s:
            f = f.next
            s = s.next
        return f
    else:
        return None


class StackWithMax():

    def __init__(self):
        self.max = [float('-infinity')]
        self.arr = []

    def _pop(self):
        p = self.arr.pop()
        if self.max[-1]==p:
            self.max.pop()
        return p

    def _push(self,value:int):
        if value>=self.max[-1]:
            self.max.append(value)
        self.arr.append(value)

    def empty(self):
        return self.arr==[]

    def _max(self):
        return self.max[-1]
    def __str__(self):
        return str(self.arr)

# st = StackWithMax()
# print(st.empty())
# st._push(3)
# st._push(4)
# st._push(5)
# st._push(1)
# print(st)
# print(st._max())
# st._pop()
# st._pop()
# st._pop()
# print(st)
# print(st._max())
# print(st.empty())
# print(st)

class BinaryTreeNode:

    def __init__(self,data=0,left=None,right=None):
        self._data = data
        self._left = left
        self._right = right

def btreeToarray(tree:BinaryTreeNode)->list:
    q1 = Queue()
    q1.enqueue(tree)
    q2 = Queue()
    arr = [[tree]]
    sub_arr = []
    while not q1.empty and not q2.empty:
        tn = q1.enqueue()
        if tn.left:
            q2.enqueue(tn.left)
            sub_arr.append(tn.left.data)
        if tn.right:
            q2.enqueue(tn.right)
            sub_arr.append(tn.right.data)
        if q1.empty:
            q1 = q2
            if sub_arr!=[]:
                arr.append(sub_arr)
                sub_arr = []
    return arr


def isBalanced(root:BinaryTreeNode)->bool:

    def findMaxHeight(root:BinaryTreeNode,hmax:int=-1)->int:
        if root:
            hmax += 1
            h1,flag1 = findMaxHeight(root.left,hmax)
            h2,flag2 = findMaxHeight(root.right,hmax)
            hmax = max(h1,h2)
            flag = flag1 and flag2
            flag = abs(h1-h2)<=1 and flag
        return hmax,flag

    left_branch = root.left
    right_branch = root.right
    left_height,flag1 = findMaxHeight(left_branch)
    right_height,flag2 = findMaxHeight(right_branch)
    flag = flag1 and flag2
    flag = abs(left_height-right_height)<=1 and flag
    return flag

def findLCA(node1 : BinaryTreeNode, node2 : BinaryTreeNode)-> BinaryTreeNode:

    if h1==h2:
        return h1
    #Find height of each node

    def find_height(node:BinaryTreeNode)-> int:
        h = 0
        while node.parent:
            h += 1
            node = node.parent
        return h

    h1,h2 = find_height(node1),find_height(node2)

    if h1<h2:
        h2,h1 = h2,h1
        node1,node2 = node2,node1
    dh = h1-h2
    for _ in range(dh):
        node2 = node2.parent

    while node1 is not node2:
        node1 = node1.parent
        node2 = node2.parent

    return node1


import heapq
def running_median(sequence:list)-> list:
    min_heap : List[int] = []#above median
    max_heap : List[int] = []#below median

    result = []

    for x in sequence:
        heapq.heappush(max_heap,-heapq.heappushpop(min_heap,x))
        print(max_heap,min_heap)
        if len(max_heap)>len(min_heap):
            heapq.heappush(min_heap,-heapq.heappop(max_heap))
        print(max_heap,min_heap)
        result.append(0.5*(min_heap[0]-max_heap[0])if len(min_heap)==len(max_heap) else min_heap[0])

    return result

#print(running_median([1,0,3,5,2,0,1]))


def bsearch(t:int,arr:list)-> int:
    start:int = 0
    end:int = len(arr)-1


    while start<=end:
        mid = start + (end-start)//2

        val = arr[mid]

        if val == t:
            return mid
        elif val < t:
            start = mid +1

        else :
            end = mid - 1


    return False

import bisect

# def search_sorted_first(arr:list,value:int)->int:
#     i = bisect.bisect_left(arr,value)
#     if i<len(arr) and arr[i] == value:
#         return i
#     return -1

def search_sorted_first(arr:list,value:int)->int:

    k = bsearch(value,arr)

    if not k:
        return -1

    while arr[k]==value:
        k -=1
    k+=1

    return k





# print(search_sorted_first([-14,-10,2,108,108,243,285,285,285,401],108))
# print(search_sorted_first([-14,-10,2,108,108,243,285,285,285,401],285))
# print(search_sorted_first([-14,-10,2,108,108,243,285,285,285,401],403))

def nearest_sqrt(n:int)->int:

    # sq = math.sqrt(n)
    # return int(sq)
    s = 0
    e = n
    mid = s + (e-s)//2
    while not (mid*mid<n and n<(mid+1)*(mid+1)):

        if mid*mid < n:
            s = mid + 1
        else :
             e = mid - 1

        mid = s + (e-s)//2
    return mid

def nearest_sqrt2(k:int)->int:

    l,r=0,k

    while l<=r :
        mid= (l+r)//2
        ms = mid*mid
        if ms<=k:
            l=mid+1
        else:
            r=mid-1
    return mid-1

num = 300
# print(nearest_sqrt(num))
# print(nearest_sqrt2(num))
import collections

def anonymous_letter(letter:str,magazin:str)->bool:
    d_letter = collections.Counter(letter)


    for c in magazin:
        if d_letter.get(c):
            d_letter[c] -= 1
            if d_letter[c] == 0:
                del d_letter[c]
                if d_letter == {}:
                    return True

    return not d_letter

# print(anonymous_letter('abcdab','abcsdlfzefx'))
# print(anonymous_letter('abcdab','abcsdlfzefxavb'))

class ISBN:

    LRU = 10
    def __init__(self):
        self.cache = {}  #{'123456789X':12}ISBN : price
        self.order = Queue()

    def insert(self,ISBN:str,price:int):
        if self.cache.get(ISBN):
            self.put_on_top(ISBN)
        else:
            self.cache[ISBN] = price

    def lookup(self,ISBN:str)->int:
        if self.cache.get(ISBN):
            self.put_on_top(ISBN)
            return self.cache[ISBN]
        else:
            return -1

    def erase(self,ISBN:str)->bool:
        if self.cache.get(ISBN):

            del self.cache[ISBN]
            return True
        else:
            return False

    def put_on_top(self,ISBN:str):
        self.order.enqueue(ISBN)
        if len(self.order)>self.LRU:
            old_ISBN = self.order.dequeu()
            del self.cache[old_ISBN]

def find_closest(arr:list)->int:
    d = {}
    closest = float('inf')
    for i in range(len(arr)):
        if d.get(arr[i]):
            if i-d[arr[i]]<closest:
                closest = i - d[arr[i]]
                pair = [arr[i],d[arr[i]],i]

        d[arr[i]] = i

    return pair

#print(find_closest(['All','work','and','no','play','makes','for','no','work','no','fun','and','no','results']))


def find_if_palyndrom(word:str)->bool:

    dic = {}
    for char in word:
        if dic.get(char):
            del dic[char]
        else:
            dic[char] = 1

    return len(dic)<=1

# print(find_if_palyndrom('edifiefa'))
# print(find_if_palyndrom('edified'))

def intersection_of_sorted_arrays(arr1:list,arr2:list)->list:

    i,j = 0,0
    result = []
    while j<len(arr2):
        while i<len(arr1) and arr1[i]<=arr2[j]:
            if arr1[i] == arr2[j]:
                result.append(arr1[i])
                while i+1<len(arr1) and arr1[i+1] == arr2[j]:
                    i += 1
            i += 1
        j += 1
    return result

def intersection_of_sorted_arrays2(arr1:list,arr2:list)->list:
    return [a for i, a in enumerate(arr1) if (i!=0 or a!=arr1[i-1]) and a in arr2]

# print(intersection_of_sorted_arrays([2,3,3,5,5,6,7,7,8,12,12],[5,5,6,8,8,9,10,10,12]))



def merge_sorted_in_place(arr1:list,arr2:list)->list:
    i,j = 0,0
    insert_index = len(arr1)-1

    while j<len(arr2):
        while i<len(arr1) and arr1[i]<=arr2[j]:
            if arr1[i] == arr2[j]:
                arr1[insert_index] = arr1[i]
                insert_index -= 1
            i += 1
        j += 1

    insert_index = len(arr1)-1
    for i in range(insert_index//2):
        arr1[i],arr[insert_index-i] = arr[insert_index-i],arr1[i]

    return arr1
#print(merge_sorted_in_place([3,13,17,None,None,None,None,None],[3,7,11,19]))
tree = BinaryTreeNode(10,BinaryTreeNode(7,BinaryTreeNode(6,None,None),BinaryTreeNode(8,None,None)),BinaryTreeNode(12,BinaryTreeNode(11,None,None),BinaryTreeNode(13,None,None)))
def is_bst(tree:BinaryTreeNode)->bool:
    tree_arr = []
    def in_order(tree:BinaryTreeNode)->list:
        if tree:
            in_order(tree._left)
            tree_arr.append(tree._data)

            in_order(tree._right)

    in_order(tree)
    for i in range(1,len(tree_arr)):
        if tree_arr[i]<tree_arr[i-1]:
            return False

    return True

#print(is_bst(tree))

def find_first_greatest_key(tree:BinaryTreeNode,key)->int:
    tree_arr = []

    def in_order(tree:BinaryTreeNode)->None:
        if tree:
            in_order(tree._left)
            tree_arr.appen(tree._data)
            in_orfer(tree._right)

    in_order(tree)

    for i in range(len(tree_arr)):
        if tree_arr[i]>key:
            return tree_arr[i]

    return -1


def find_first_greatest_key_logn(tree:BinaryTreeNode,key)->int:

    best_value = float('infinity')

    def bst_traversal(tree:BinaryTreeNode,best:int)->None:

        if tree:
            if tree._data <= key:
                #best_value = tree._data
                best = bst_traversal(tree._right,best)

            else :
                if tree._data < best :
                    best = tree._data
                best = bst_traversal(tree._left,best)
        return best

    best_value = bst_traversal(tree,best_value)
    if best_value == float('infinity'):
        best_value = -1

    return best_value
#
# print(find_first_greatest_key_logn(tree,12))
# print(find_first_greatest_key_logn(tree,6))
# print(find_first_greatest_key_logn(tree,19))


def find_k_largest(tree:BinaryTreeNode, k:int)->list:
    result = []
    count = k
    def in_order_inversed(tree:BinaryTreeNode)->None:
        nonlocal count
        if tree and count >=0:
            count -=1
            in_order_inversed(tree._right)
            result.append(tree._data)
            in_order_inversed(tree._left)

    in_order_inversed(tree)

    # if count !=0 :
    #     return 'error'

    return result

#print(find_k_largest(tree,3))


def hanoi_tower(n:int,_from:int=0,_to:int=1,_buffer:int=2)->list:
    def swap(hanoi:list,f:int,b:int,t:int)->list:
        tower_h = hanoi.copy()

        for i in range(len(ht)):
            for j in range(2):
                if tower_h[i][j] == f:
                    tower_h[i][j] = b
                if tower_h[i][j] == b:
                    tower_h[i][j] = t
                if tower_h[i][j] == t:
                    tower_h[i][j] = f
        return tower_h

    if n==1:
        return [[_from,_to]]

    else :
        ht = hanoi_tower(n-1,_from,_buffer,_to)
        th = swap(ht,_from,_buffer,_to)
        return ht+[[_from,_to]]+th

# print(hanoi_tower(2))
# print(hanoi_tower(3))
# print(hanoi_tower(6))
# print(list(reversed(range(1,4+1)))+[[]for _ in range(3)])

def queens_placement(n:int)-> list:
    col_placement = []

    def solve_queens(row):
        if row == n:
            result.append(col_placement.copy())
            return
        for col in range(n):
            if all(abs(c-col) not in (0,row-i) for i,c in enumerate(col_placement[:row])):
                col_placement[row] = col

def array_permutation(arr:list)->list:

    def perms(n,sub_arr):
        recursive_result = []
        if sub_arr==[]:
            return [[n]]
        else :
            for j in range(len(sub_arr)):
                for i in range(len(sub_arr[j])+1):
                    if i == 0:
                        recursive_result.append([n]+sub_arr[j])
                    elif i == len(sub_arr[j]):
                        recursive_result.append(sub_arr[j]+[n])
                    else :
                        recursive_result.append(sub_arr[j][:i]+[n]+sub_arr[j][i:])
        return recursive_result


    result = []
    for i in range(len(arr)):
        result = perms(arr[i],result)

    return result

# print(array_permutation([1,2]))
# print(array_permutation([1,2,3]))
# print(array_permutation([1,2,3,4]))


# def score_combination(score:int)->list:
#     cache_score = {'2':[[2]],'3':[[3]],'7':[[7],[2,2,3]]}
#
#     def dp_score(sub_score):
#
#         if cache_score.get(str(sub_score)):
#             return cache_score[str(sub_score)]
#
#         if sub_score<=1:
#             return None
#
#
#         possible_score = [2,3,7]
#         for i in possible_score:
#             if sub_score>=i :
#                 r = dp_score(sub_score-i)
#
#                 if r :
#                     arr = r.copy()
#                     for j in range(len(arr)):
#                         if arr[j]:
#                             arr[j] = arr[j]+[i]
#
#
#                     if cache_score.get(str(sub_score)):
#                         if all([len(arr[0][:])!=len(c) for c in cache_score[str(sub_score)]]):
#                             cache_score[str(sub_score)].append(arr[0][:])
#                     else:
#                         cache_score[str(sub_score)] = arr[:]
#
#
#
#
#
#         return list(cache_score[str(sub_score)])
#
#
#     dp_score(score)
#     if cache_score.get(str(score)):
#         return cache_score[str(score)]
#     else:
#         return []


def score_combination(score:int)->int:

    cache_score = {'2':1,'3':1,'7':2}

    def dp_score(sub_score):
        if cache_score.get(str(sub_score)):
            return cache_score[str(sub_score)]

        if sub_score<=1:
            return None


        possible_score = [2,3,7]
        for i in possible_score:
            if sub_score>=i :
                r = dp_score(sub_score-i)
                if r :
                    r = r + 1
                    if cache_score.get(str(sub_score)):
                        cache_score[str(sub_score)] += 1
                    else:
                        cache_score[str(sub_score)] = 1
        print(cache_score)
        return cache_score[str(sub_score)]

    dp_score(score)
    if cache_score.get(str(score)):
        return cache_score[str(score)]
    else:
        return 0


# print(score_combination(6))
# print(score_combination(12))


def has_three_sum(arr:list,t:int)->bool:
    arr.sort()
    #2,3,5,7,11
    for i in range(len(arr)):
        sub_t = t - arr[i]

        j,k = 0,len(arr)-1
        while j<=k:
            if arr[j]+arr[k]==sub_t:
                return True
            elif arr[j]+arr[k]>sub_t:
                k-=1
            else:
                j+=1
    return False

# print(has_three_sum([11,2,5,7,3],21))
# print(has_three_sum([11,2,5,7,3],22))
import collections

def find_maze_exit(maze:list,s:list,e:list)->list:
    graph = collections.defaultdict(list)

    def build_maze_graph(maze):
        x_max = len(maze[0])-1
        y_max = len(maze)-1

        for i in range(len(maze)):
            for j in range(len(maze)):
                if graph.get(j):
                    graph[j][i] = []
                else:
                    graph[j] = {i:[]}
                if i<x_max:
                    graph[j][i].append([i+1,j,maze[j][i+1]])
                if j<y_max:
                    graph[j][i].append([i,j+1,maze[j+1][i]])
                if i>0:
                    graph[j][i].append([i-1,j,maze[j][i-1]])
                if j>0:
                    graph[j][i].append([i,j-1,maze[j-1][i]])

    def dfsearch(graph,curr,e):
        if curr[:2] == e:
            return True

        elif curr in visited:
            return False

        visited.append(curr)

        next = graph[curr[1]][curr[0]]

        for i in range(len(next)):

            if next[i][2]!='BLACK' and dfsearch(graph,next[i],e):
                way.append(next[i][:2])
                print(way)
                return True

        return False

    build_maze_graph(maze)
    visited = []
    way = []


    dfsearch(graph,s,e)

    return way[::-1]

mini_maze=[['WHITE','WHITE','WHITE','WHITE','WHITE'],['WHITE','WHITE','WHITE','WHITE','WHITE'],['WHITE','WHITE','WHITE','WHITE','WHITE'],['WHITE','WHITE','WHITE','WHITE','WHITE'],['WHITE','WHITE','WHITE','WHITE','WHITE']]
mini_maze2=[['WHITE','BLACK','WHITE','WHITE','WHITE'],['WHITE','BLACK','WHITE','BLACK','WHITE'],['WHITE','BLACK','WHITE','BLACK','WHITE'],['WHITE','BLACK','WHITE','BLACK','WHITE'],['WHITE','WHITE','WHITE','BLACK','WHITE']]

#print(find_maze_exit(mini_maze2,[0,0],[4,4]))


def dict_BFS(D:dict,s:str,t:str)->int:

    def build_graph(D):
        graph = {}
        for i in range(len(D)):
            for j in range(len(D)):
                if i!=j:
                    word1 = D[i]
                    word2 = D[j]
                    differente_letters = 0
                    for k in range(len(word1)):
                        if word1[k]!=word2[k]:
                            differente_letters += 1
                    if differente_letters<=1:
                        if graph.get(word1):
                            graph[word1].append(word2)
                        else:
                            graph[word1] = [word2]
        return graph

    # def bfs(graph,s,t):
    #     if s == t:
    #         return True
    #     elif visited_s.get(t):
    #         return True
    #     elif visited_t.get(s):
    #         return True
    #     elif visited_s.get(s):
    #         return False
    #     elif visited_t.get(t):
    #         return False
    #
    #
    #
    #     while qs and qt:
    #         next_s = qs.pop(0)
    #         next_t = qt.pop(0)
    #         visited_s[next_s] = True
    #         visited_t[next_t] = True
    #         if graph.get(next_s):
    #             for i in graph[next_s]:
    #                 if not visited_s.get(i):
    #                     qs.append(i)
    #         else:
    #             return False
    #         if graph.get(next_t):
    #             for i in graph[next_t]:
    #                 if not visited_t.get(i):
    #                     qt.append(i)
    #
    #         else:
    #             return False
    #         bs = bfs(graph,next_s,t)
    #         if bs:
    #             path.append(next_s)
    #         bt = bfs(graph,s,next_t)
    #         if bt:
    #             path.append(next_t)
    #         print(qs,qt)
    #     return False

    def bfs(graph,s,t):
        qs = []
        qs.append([s,1])
        visited_s = {}
        if s == t:
            num_of_path = 1


        while qs:
            next_word = qs.pop(0)
            count = next_word[1]
            next_word = next_word[0]
            visited_s[next_word] = True

            for word in graph[next_word]:
                if not visited_s.get(word):

                    qs.append([word,count+1])
            if next_word==t:
                return count


        return -1





    graph = build_graph(D)




    return bfs(graph,s,t)


#print(dict_BFS(['bat','cot','dog','dag','dot','cat'],'cat','dog'))


def paint_matrix(A:list,c:list)->list:

    x = c[0]
    y = c[1]

    x_max = len(A[0])-1
    y_max = len(A)-1

    color = A[y][x]

    q = [[x,y]]
    visited = {}
    while q:
        coord = q.pop(0)
        x = coord[0]
        y = coord[1]

        if A[y][x] == color:
            A[y][x] = not color

            if x+1<=x_max and A[y][x+1] == color:
                q.append([x+1,y])
            if x-1>=0 and A[y][x-1] == color:
                q.append([x-1,y])
            if y+1<=y_max and A[y+1][x] == color:
                q.append([x,y+1])
            if y-1>=0 and A[y-1][x] == color:
                q.append([x,y-1])

    return A

mini_canva = [[False,False,False,False],[False,False,False,False],[True,True,True,True],[False,False,False,False]]
#print(paint_matrix(mini_canva,[0,0]))

def test():
    x=1
    y=1
    for [i,j] in [[x+1,y],[x-1,y],[x,y+1],[x,y-1]]:
        print(i)
        print(j)

test()
