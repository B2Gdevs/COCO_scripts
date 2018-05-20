'''
This script will show the difference between using a list comprhension and 
for loops.  For speed just use list comprehensions as much as possible. 
They are the fastest is one MUST use a list.

Examples are in descending order from fastest to slowest.
'''

import time

a = [1, 2, 3, 4, 5]

start = time.time()


'''
This is a nested for loop but as a list comprehension
'''
b = [value if not value % 2 ==1 else value * 1 for i in range(0,1000000) for value in a]

end = time.time()

print(end - start)

start = time.time()
'''
This is a for loop with a list comprehension in it that does the same thing
as the first example
'''
for i in range(0, 1000000):

    c = [value if not value % 2 == 1 else value * 1 for value in a]

end = time.time()

print(end - start)

start = time.time()

'''
This is the basic nested for loop that does the same thing as the previous 
examples
'''
for i in range(0, 1000000):

    for j in range(len(a)):

        if not a[j] % 2 == 1:

            a[j] = a[j]

        else:

            a[j] = a[j] * 1

end = time.time()

print(end - start)