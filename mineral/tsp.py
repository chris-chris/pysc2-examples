import doctest
from itertools import permutations


def distance(point1, point2):
  """
  Returns the Euclidean distance of two points in the Cartesian Plane.

  >>> distance([3,4],[0,0])
  5.0
  >>> distance([3,6],[10,6])
  7.0
  """
  return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) ** 0.5


def total_distance(points):
  """
  Returns the length of the path passing throught
  all the points in the given order.

  >>> total_distance([[1,2],[4,6]])
  5.0
  >>> total_distance([[3,6],[7,6],[12,6]])
  9.0
  """
  return sum([distance(point, points[index + 1]) for index, point in enumerate(points[:-1])])


def travelling_salesman(points, start=None):
  """
  Finds the shortest route to visit all the cities by bruteforce.
  Time complexity is O(N!), so never use on long lists.

  >>> travelling_salesman([[0,0],[10,0],[6,0]])
  ([0, 0], [6, 0], [10, 0])
  >>> travelling_salesman([[0,0],[6,0],[2,3],[3,7],[0.5,9],[3,5],[9,1]])
  ([0, 0], [6, 0], [9, 1], [2, 3], [3, 5], [3, 7], [0.5, 9])
  """
  if start is None:
    start = points[0]
  return min([perm for perm in permutations(points) if perm[0] == start], key=total_distance)


def optimized_travelling_salesman(points, start=None):
  """
  As solving the problem in the brute force way is too slow,
  this function implements a simple heuristic: always
  go to the nearest city.

  Even if this algoritmh is extremely simple, it works pretty well
  giving a solution only about 25% longer than the optimal one (cit. Wikipedia),
  and runs very fast in O(N^2) time complexity.

  >>> optimized_travelling_salesman([[i,j] for i in range(5) for j in range(5)])
  [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 4], [1, 3], [1, 2], [1, 1], [1, 0], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [3, 4], [3, 3], [3, 2], [3, 1], [3, 0], [4, 0], [4, 1], [4, 2], [4, 3], [4, 4]]
  >>> optimized_travelling_salesman([[0,0],[10,0],[6,0]])
  [[0, 0], [6, 0], [10, 0]]
  """
  if start is None:
    start = points[0]
  must_visit = points
  path = [start]
  must_visit.remove(start)
  while must_visit:
    nearest = min(must_visit, key=lambda x: distance(path[-1], x))
    path.append(nearest)
    must_visit.remove(nearest)
  return path

def main():
  doctest.testmod()
  points = [[0, 0], [1, 5.7], [2, 3], [3, 7],
            [0.5, 9], [3, 5], [9, 1], [10, 5]]
  print("""The minimum distance to visit all the following points: {}
starting at {} is {}.

The optimized algoritmh yields a path long {}.""".format(
    tuple(points),
    points[0],
    total_distance(travelling_salesman(points)),
    total_distance(optimized_travelling_salesman(points))))


if __name__ == "__main__":
  main()