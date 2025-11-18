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
        self.count = 0 # Unique tie-breaker counter

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
        self.count += 1

        entry = (priority, self.count, item)
        self.heap.append(entry) # Add new element at the end
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
            return self.heap.pop()[2] # Returns the item only, not the priority

        root_item = self.heap[0][2] # Store the root item

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


def algorithm_a_star(draw_callback, grid, start, goal):
    """
    A* Algorithm
    Uses Custom Priority Queue Data Structure
    """

    frontier = PriorityQueue()

    # Push (f_cost, node) to the priority queue.

    # Start node initialization
    start.g_cost = 0
    start.h_cost = heuristic(start.get_pos(), goal.get_pos())
    start.f_cost = start.g_cost + start.h_cost

    frontier.push(start.f_cost, start)

    explored_set = {start}

    while not frontier.is_empty():
        # Check for quit events to avoid freezing the program
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
       
        # Get the node with the lowest f_cost
        current_node = frontier.pop()

        if current_node in explored_set:
            explored_set.remove(current_node)
        else:
            continue # Skip if already explored

        # Check for goal
        if current_node == goal:
            reconstruct_path(goal, draw_callback)
            goal.make_end() # Redraw the end node
            start.make_start() # Redraw the start node
            return True
        
        # Explore neighbors
        for neighbor in current_node.neighbors:
            # Calculate g_cost
            temp_g_cost = current_node.g_cost + 1

            # If this path to neighbor is better than any other
            if temp_g_cost < neighbor.g_cost:
                # Update neighbor's values
                neighbor.parent = current_node
                neighbor.g_cost = temp_g_cost
                neighbor.h_cost = heuristic(neighbor.get_pos(), goal.get_pos())
                neighbor.f_cost = neighbor.g_cost + neighbor.h_cost

                # Add to frontier if not there
                if neighbor not in explored_set:
                    frontier.push(neighbor.f_cost, neighbor)
                    explored_set.add(neighbor)
                    neighbor.make_open() # Mark as in frontier
        
        draw_callback() # Update visualization

        if current_node != start:
            current_node.make_closed() # Mark as explored
    return False # Path not found

def main(win, width):
    ROWS = 50
    grid = make_grid(ROWS, width)
    start_node = None
    end_node = None
    run = True
    started = False

    font = pygame.font.SysFont('Arial', 18)
    instructions = [
        "Left Click: Draw Start (Green), End (Red), and Walls (Black)",
        "Right Click: Remove Node",
        "---",
        "B: Run BFS  |  D: Run DFS  | A: Run A* Search",
        "C: Clear Grid"
    ]

    new_height = width + (len(instructions) * 20) + 10
    win = pygame.display.set_mode((width, new_height))

    while run:
        draw(win, grid, ROWS, width)

        panel_y = width 
        pygame.draw.rect(win, COLOR_BLACK, (0, panel_y, width, new_height - width))
        for i, line in enumerate(instructions):
            text = font.render(line, True, COLOR_WHITE)
            win.blit(text, (10, panel_y + 5 + i * 20))
        pygame.display.update()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False

            if started:
                continue

            if pygame.mouse.get_pressed()[0]: # Left Click
                pos = pygame.mouse.get_pos()
                if pos[1] < width:
                    row, col = get_clicked_position(pos, ROWS, width)
                    node = grid[row][col]

                    if not start_node and node != end_node:
                        start_node = node
                        start_node.make_start()
                    
                    elif not end_node and node != end_node:
                        end_node = node
                        end_node.make_end()
                    
                    elif node != end_node and node != start_node:
                        node.make_wall()
            
            elif pygame.mouse.get_pressed()[2]: # Right Click
                pos = pygame.mouse.get_pos()
                if pos[1] < width:
                    row, col  = get_clicked_position(pos, ROWS, width)
                    node = grid[row][col]
                    node.reset()
                    if node == start_node:
                        start_node = None
                    elif node == end_node:
                        end_node = None
            
            if e.type == pygame.KEYDOWN:
                if (e.key == pygame.K_b or e.key == pygame.K_d or e.key == pygame.K_a) and start_node and end_node:
                    started = True
                    for row in grid:
                        for node in row:
                            node.update_neighbors(grid)
                            node.g_cost = float("inf")
                            node.h_cost = float("inf")
                            node.f_cost = float("inf")
                    
                    draw_callback = lambda: draw(win, grid, ROWS, width)

                    if e.key == pygame.K_b:
                        algorithm_bfs(draw_callback, grid, start_node, end_node)
                    elif e.key == pygame.K_d:
                        algorithm_dfs(draw_callback, grid, start_node, end_node)
                    elif e.key == pygame.K_a:
                        algorithm_a_star(draw_callback, grid, start_node, end_node)
                    
                    started = False

                if e.key == pygame.K_c:
                    start_node = None
                    end_node = None
                    grid = make_grid(ROWS, width)
                    started = False
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    pygame.init()
    main(WIN, WIDTH)