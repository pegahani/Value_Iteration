# Queues: Stack, FIFOQueue, PriorityQueue
import bisect
import numpy as np

ftype = np.float32

def update(x, **entries):
    """Update a dict; or an object with slots; according to entries."""
    """>>> update({'a': 1}, a=10, b=20)"""
    """{'a': 10, 'b': 20}"""
    """>>> update(Struct(a=1), a=10, b=20)"""
    """Struct(a=10, b=20)
    """
    if isinstance(x, dict):
        x.update(entries)
    else:
        x.__dict__.update(entries)
    return x

class Queue:
    """Queue is an abstract class/interface. There are three types:
        Stack(): A Last In First Out Queue.
        FIFOQueue(): A First In First Out Queue.
        PriorityQueue(lt): Queue where items are sorted by lt, (default <).
    Each type supports the following methods and functions:
        q.append(item)  -- add an item to the queue
        q.extend(items) -- equivalent to: for item in items: q.append(item)
        q.pop()         -- return the top item from the queue
        len(q)          -- number of items in q (also q.__len())
    Note that isinstance(Stack(), Queue) is false, because we implement stacks
    as lists.  If Python ever gets interfaces, Queue will be an interface."""

    def __init__(self):
        abstract

    def extend(self, items):
        for item in items: self.append(item)

def Stack():
    """Return an empty list, suitable as a Last-In-First-Out Queue."""
    return []

class my_data_struc(Queue):

    """A First-In-First-Out Queue."""
    def __init__(self):
        self.A = []

    def append(self, item):
        self.A.append(item)

    def __len__(self):
        return len(self.A)

    def show_A(self):
        for i in range(self.__len__()):
            print "element ", i, " = ", self.A[i].state

    def update(self, indexes):
        """
        it gets a list of indexes of A members and updates A list
        :param indexes: a list subset of A for instance if len(A) = 10, this list can be: indexes=[1,5,3,6]
        :return: it updates the current A list inside FIFQUEUE
        """
        listy = []
        for i in range(len(self.A)):
            if i in indexes:
                #self.A[i].inside_convex= True
                listy.append(self.A[i])

        self.A = listy
        return

    def pareto_comparison(self, a, b):

        a = np.array(a, dtype= ftype)
        b = np.array(b, dtype= ftype)

        assert len(a)==len(b), \
                "two vectors don't have the same size"

        return all(a>=b)

    def is_empty(self):
        return True if not self.A else False

class FIFOQueue(Queue):

    """A First-In-First-Out Queue."""
    def __init__(self):
        self.A = []; self.start = 0

    def append(self, item):
        self.A.append(item)

    def __len__(self):
        return len(self.A) - self.start

    def extend(self, items):
        self.A.extend(items)

    def pop(self):
        assert self.A ,\
                   "no element can be poped from an empty list"

        e = self.A[self.start]
        self.start += 1
        #if self.start > 5 and self.start > len(self.A)/2:
        self.A = self.A[self.start:]
        self.start = 0
        return e

    def pareto_comparison(self, a, b):

        a = np.array(a, dtype= ftype)
        b = np.array(b, dtype= ftype)

        assert len(a)==len(b), \
                "two vectors don't have the same size"

        return all(a>=b)

    def pop_priority(self):
        """
        this function choose a node in the current queue. it selects epsilon non-dominated node from queue.
        :return: it returns the node which has a non_dominated v_bar_d in comparison to rest of queue elements.
        """

        assert self.A ,\
                   "no element can be poped from an empty list"

        prefered = self.A[0].state[1]
        index = 0

        for i in range(1,len(self.A)):
            if self.pareto_comparison(self.A[i].state[1], prefered):
                index = i
                prefered = self.A[i].state[1]

        e = self.A[index]
        self.A = self.A[0:index]+self.A[index+1:]

        return e

    def is_empty(self):
        return True if not self.A else False


