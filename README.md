# Algorithm Visualizer: Search & Pathfinding (Custom DSA)

A self-development project built to deeply understand and visualize core search algorithms (Uninformed and Informed) by implementing all necessary **data structures from scratch**.

-----

## Project Overview

This application provides an interactive, visual platform to observe how fundamental pathfinding algorithms explore a 2D grid. The primary goal of this project was not just to run the algorithms, but to implement the core data structures (Queues, Stacks, and Priority Queues) that drive them, a good exercise to practice Data Structures and Algorithms (DSA) principles.

**Key Features:**

* Custom DSA Implementation: All algorithms use custom-built `Queue`, `Stack`, and `PriorityQueue` (Min-Heap) classes, avoiding Python's built-in modules.
* Visual Step-Through: Observe the "Frontier" (purple node) and "Explored Set" (orange node) as the algorithms run.
* Uninformed Search: Visualization for Breadth-First Search (BFS) and Depth-First Search (DFS).
* Informed Search: Visualization for A\* Search, utilizing the Manhattan Distance heuristic.
* Interactive World: Draw walls, set start/end points, and watch the search process solve the maze.

## Learning Outcomes and DSA Concepts

Building this visualizer from scratch cemented several core concepts essential to both DSA and AI search algorithms:

1. Search Strategy vs. Data Structure: I learned that the search algorithm's behavior is dictated entirely by its frontier structure:

  * BFS requires a Queue to ensure optimality (exploring breadth-first).
  * DFS requires a Stack to enforce depth-first exploration.

2. The Heap Invariant (A*): I gained a deep understanding of the Min-Heap property and the necessity of `_heapify_up` and `_heapify_down` functions. These are not merely helper methods; they are the sole mechanisms that ensure the lowest cost node is always at the root, which is critical for A*'s performance guarantee.

3. Heuristic Cost and Priority: The A* implementation clarified the role of the evaluation function $f(n) = g(n) + h(n)$. The search is constantly re-prioritized, not just by the distance traveled ($g$), but by the estimated distance remaining ($h$), which is the key distinction between informed and uninformed search.

4. andling Non-Comparable Types: I learned the practical solution for handling ties in a custom PriorityQueue by including a unique, monotonic `count` variable in the heap entry: (f_cost, count, node). This prevents runtime errors and ensures consistent tie-breaking.

## Installation and Setup (Assuming you already have Python 3.x installed)

### Install Pygame

```bash
python3 -m pip install pygame
```

## How to Use the Visualizer

The application opens an interactive grid window. The following controls allow you to build and solve mazes:

| Action             | Control               | Purpose                                          |
| :----------------- | :-------------------: | :----------------------------------------------- |
| Set Start Node     | Left Click (1st)      | Places the Green Start Node.                     |
| Set End Node       | Left Click (2nd)      | Places the Red End Node.                         |
| Draw Wall          | Left Click and Drag   | Creates Black Walls (Obstacles).                 |
| Erase Node         | Right Click and Drag  | Resets any node (Start, End, or Wall) to white.  |
| Run BFS            | [B]-Key               | Starts Breadth-First Search visualization.       |
| Run DFS            | [D]-Key               | Starts Depth-First Search visualization.         |
| Run A*             | [A]-Key               | Starts A* Search visualization.                  |
| Clear              | [C]-Key               | Clears the entire board and resets all nodes.    |

## Algorithm and DSA Core

The behavior of each algorithm is defined by the underlying data structure used for its Frontier (the set of nodes waiting to be explored).

| Algorithm  | Search Type       | Data Structure Used       | Principle                                                                                                           | Path Guarantee             |
|:---------- | :---------------- | :------------------------ | :------------------------------------------------------------------------------------------------------------------ | :------------------------- |
| BFS        | Uninformed Search | Queue (FIFO)              | Explores level by level, guaranteeing all nodes at depth d are checked before moving deeper.                        | Shortest Path Guaranteed   |
| DFS        | Uninformed Search | Stack (LIFO)              | Explores by diving as deep as possible down one branch before backtracking.                                         | No Shortest Path Guarantee |
| A\*        | Informed Search   | Priority Queue (Min-Heap) | Prioritizes nodes based on the lowest f_cost (g\_cost + h\_cost), ensuring efficient and goal-directed expansion. | Shortest Path Guaranteed   |

### Focus: The Custom Priority Queue

The A\* algorithm relies on a PriorityQueue implemented as a Min-Heap. This structure is critical because:

* It ensures the node with the absolute lowest estimated total cost is always expanded next.
* The heap logic (`_heapify_up` and `_heapify_down`) is implemented manually to maintain the min-heap property and ensure $O(\log N)$ push/pop efficiency.
* The `count` variable is included in the heap entry (f_cost, count, node) to consistently break ties when multiple nodes have the same cost.
