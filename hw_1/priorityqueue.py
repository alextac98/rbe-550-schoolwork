#!/usr/bin/env python3

class PriorityQueue:
    def __init__(self):
        self.queue = []
    
    def put(self, value, weight):
        self.queue.append((weight, value))
        self.queue.sort(reverse=False)

    def get(self, i: int = 0):
        return self.queue[i]

    def get_weight(self, value):
        for (v, w) in self.queue():
            if v == value:
                return w

    def empty(self):
        if len(self.queue) == 0:
            return True
        return False

    def __len__(self):
        return len(self.queue)