import pygame
import math
import sys

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Pathfinding Algorithm Visualizer Using Custom Made DSA")

COLOR_RED = (255, 0, 0)         # For End Node
COLOR_GREEN = (0, 255, 0)       # For Start Node
COLOR_BLUE = (0, 0, 255)        # For Path
COLOR_YELLOW = (255, 255, 0)    # Not used yet
COLOR_WHITE = (255, 255, 255)   # For empty Node
COLOR_BLACK = (0, 0, 0)         # For Wall/Obstacle
COLOR_GREY = (128, 128, 128)    # For Grid Lines
COLOR_PURPLE = (128, 0, 128)    # For Frontier Node
COLOR_ORANGE = (255, 165, 0)    # For Visited Node/Explored Node

class Queue:
    """
    Simple FIFO Queue implementation using a list
    Custom Data Structure for BFS Algorithm
    """

    def __init__(self):
        self.items = []

    def enqueue(self, item):
        """
        Add an item to the end of the queue
        """
        self.items.append(item)

    def dequeue(self):
        """
        Removes and returns the item from the front of the queue
        """
        if not self.is_empty():
            return self.items.pop(0)
        return None
    
    def is_empty(self):
        """
        Returns True if the queue is empty.
        """
        return len(self.items) == 0
    
    def view_queue(self):
        """
        Returns the current items in the queue
        """
        return self.items
    
class Stack:
    """
    Simple LIFO Stack implementation using a list 
    Custom Data Structure for DFS Algorithm
    """
    def __init__(self):
        self.items = []

    def push(self, item):
        """
        Add an item to the top of the stack
        """
        self.items.append(item)
    
    def pop(self):
        """
        Removes and returns the item from the top of the stack
        """
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def is_empty(self):
        """
        Returns True if the stack is empty.
        """
        return len(self.items) == 0
    
    def peek(self):
        """
        Returns the top item of the stack without removing it
        """
        if not self.is_empty():
            return self.items[-1]
        return None
    
    def view_stack(self):
        """
        Returns the current items in the stack
        """
        return self.items
    
class PriorityQueue:
    """
    A Min-Heap implementation
    Custom Data Structure for A*, UCS, and Greedy Best-First.
    Stores items as (priority, item) tuples.
    """
    def __init__(self):
        self.heap = []

    def is_empty(self):
        """
        Returns True if the priority queue is empty.
        """
        return len(self.heap) == 0
    
    def push(self, priority, item):
        """
        Adds a new item to the heap and heaps up to maintain the 
        min-heap property.
        """
        self.heap.append((priority, item)) # Add new element at the end
        self._heapify_up(len(self.heap) - 1) # Heap up to correct position

    def pop(self):
        """
        Removes and returns the item with the highest priority (lowest value).
        Heaps down to maintain the min-heap property.
        """
        if self.is_empty():
            return None
        
        # If only one element, just pop and return it
        if len(self.heap) == 1:
            return self.heap.pop()[1] # Returns the item only, not the priority

        root_item = self.heap[0][1] # Store the root item

        self.heap[0] = self.heap.pop() # Replace root with last element

        self._heapify_down(0) # Heap down to correct position

        return root_item
    
    def _heapify_up(self, index):
        """
        Moves an item at 'index' up the heap until it's in the 
        correct spot meaning it's smaller than its chidren 
        and larger than its parent.
        """
        parent_index = (index - 1) // 2

        # While we are not at the root (index > 0) and
        # our item has a lower priority than its parent
        while index > 0 and self.heap[index][0] < self.heap[parent_index][0]:
            # Swap the current item with its parent
            self._swap(index, parent_index)

            # Move up to the parent's position
            index = parent_index
            parent_index = (index - 1) // 2

    def _heapify_down(self, index):
        """
        Moves an item at 'index' down the heap until it's in the
        correct spot.
        """
        last_index = len(self.heap) - 1

        while True:
            left_child_index = 2 * index + 1
            right_child_index = 2 * index + 2
            smallest_child_index = index # Assume parent is smallest

            # Check if left child exists and is smaller than parent
            if (left_child_index <= last_index and
                self.heap[left_child_index][0] < self.heap[smallest_child_index][0]):
                smallest_child_index = left_child_index

            # Check if right child exists and is smaller than current smallest
            if (right_child_index <= last_index and
                self.heap[right_child_index][0] < self.heap[smallest_child_index][0]):
                smallest_child_index = right_child_index
            
            # If the smallest child is not the parent, swap them
            if smallest_child_index != index:
                self._swap(index, smallest_child_index)
                # Move down to the child's position
                index = smallest_child_index
            else:
                # We are in the correct spot
                break
    
    def _swap(self, i, j):
        """
        Helper function to swap two items in the heap.
        """
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]    

class Node:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = COLOR_WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows
        self.parent = None # For path reconstruction

        # For A* and UCS
        self.g_cost = float("inf")  # Cost from start node
        self.h_cost = float("inf")  # Heuristic cost to end node
        self.f_cost = float("inf")  # Total cost (g + h)

    def get_pos(self):
        return self.row, self.col
    
    def is_closed(self):
        return self.color == COLOR_ORANGE
    
    def is_open(self):
        return self.color == COLOR_PURPLE
    
    def is_wall(self):
        return self.color == COLOR_BLACK
    
    def is_start(self):
        return self.color == COLOR_GREEN
    
    def is_end(self):
        return self.color == COLOR_RED
    
    def reset(self):
        self.color = COLOR_WHITE
        self.parent = None
        self.g_cost = float("inf")
        self.h_cost = float("inf")
        self.f_cost = float("inf")
    
    def make_start(self):
        self.color = COLOR_GREEN
    
    def make_closed(self):
        self.color = COLOR_ORANGE
    
    def make_open(self):
        self.color = COLOR_PURPLE
    
    def make_wall(self):
        self.color = COLOR_BLACK
    
    def make_end(self):
        self.color = COLOR_RED
    
    def make_path(self):
        self.color = COLOR_BLUE
    
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))
    
    def update_neighbors(self, grid):
        """
        Finds all valid, non-wall neighbors (up, down, left, right)
        """
        self.neighbors = []
        # Down
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_wall():
            self.neighbors.append(grid[self.row + 1][self.col])
        
        # Up
        if self.row > 0 and not grid[self.row - 1][self.col].is_wall():
            self.neighbors.append(grid[self.row - 1][self.col])

        # Right
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_wall():
            self.neighbors.append(grid[self.row][self.col + 1])
        
        # Left
        if self.col > 0 and not grid[self.row][self.col - 1].is_wall():
            self.neighbors.append(grid[self.row][self.col - 1])

# Helper Functions 

def heuristic(p1, p2):
    """
    Heuristic function Manhattan distance
    """
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def make_grid(rows, width):
    grid = []
    gap = width // rows

    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(i, j, gap, rows)
            grid[i].append(node)
    return grid

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, COLOR_GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, COLOR_GREY, (j * gap, 0), (j * gap, width))

def draw(win, grid, rows, width):
    win.fill(COLOR_WHITE)

    for row in grid:
        for node in row:
            node.draw(win)
    
    draw_grid(win, rows, width)
    pygame.display.update()

def get_clicked_position(pos, rows, width):
    gap = width // rows
    y, x = pos

    row = y // gap
    col = x // gap

    return row, col

def reconstruct_path(end_node, draw_callback):
    """
    Backtracks from the end_node via its parent
    to draw the final path.
    """
    current = end_node
    while current.parent:
        current = current.parent

        if not current.is_start(): # To avoid changing start node color
            current.make_path()
        draw_callback()
        pygame.time.delay(40) # Slow down for visualization


def algorithm_bfs(draw_callback, grid, start, goal):
    """
    Breadth-First Search Algorithm
    Uses Custom Queue Data Structure
    """
    frontier = Queue() # Create the frontier queue
    frontier.enqueue(start) # Add the start node

    explored_set = {start} # To keep track of explored nodes

    while not frontier.is_empty():
        # Check for quit events to avoid freezing the program
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        current_node = frontier.dequeue() # Get the next node from the front of the queue

        # Check if current_node is the goal node
        if current_node == goal:
            goal.parent = current_node.parent # Link the goal node
            reconstruct_path(goal, draw_callback)
            goal.make_end() # Redraw the end node
            start.make_start() # Redraw the start node
            return True # Path found
        
        # Explore neighbors
        for neighbor in current_node.neighbors:
            if neighbor not in explored_set:
                # Add neighbor to the explored set and frontier
                explored_set.add(neighbor)
                neighbor.parent = current_node # Set parent for path reconstruction
                frontier.enqueue(neighbor)
                neighbor.make_open() # Mark as in frontier

        draw_callback() # Update visualization

        # Mark current node as explored
        if current_node != start:
            current_node.make_closed()
        
    return False # Path not found


def algorithm_dfs(draw_callback, grid, start, goal):
    """
    Depth-First Search Algorithm
    Uses Custom Stack Data Structure
    """
    
    frontier = Stack() 
    frontier.push(start)

    # Add to visited either while popping from stack
    # OR when pushing to stack... Same thing
    explored_set = set()

    while not frontier.is_empty():
        # Check for quit events to avoid freezing the program
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        current_node = frontier.pop()

        # Check if current_node is the goal node
        if current_node == goal:
            goal.parent = current_node.parent # Link the goal node
            reconstruct_path(goal, draw_callback)
            goal.make_end() # Redraw the end node
            start.make_start() # Redraw the start node
            return True # Path found
        
        # If we haven't visited this node yet
        if current_node not in explored_set:
            explored_set.add(current_node) # Mark as visited

            if current_node != start:
                current_node.make_closed() # Mark as explored

            
            # Add the neighbors to the stack
            for neighbor in current_node.neighbors:
                if neighbor not in explored_set:
                    neighbor.parent = current_node
                    frontier.push(neighbor)
                    neighbor.make_open() # Mark as in frontier
        
        draw_callback() # Update visualization
    
    return False # Path not found




