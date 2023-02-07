import pickle

from main import num_iterations
import maze

"""Stand alone script to generate new mazes to mazes.pickles, run by main on each run"""

num_iterations = num_iterations #number of mazes to create

maze_cache = []
print("Generating Mazes:")
for j in range(num_iterations):
    print(f"Starting maze {j}")
    maze_cache.append(maze.make_maze())
print("Done!")

#output to file
with open('mazes.pickle', 'wb') as handle:
    pickle.dump(maze_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)