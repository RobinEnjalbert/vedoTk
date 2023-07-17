from os.path import join
from os import remove
from vedoTk import MeshSelection

# Selection from scratch
mesh_file = join('resources', 'armadillo.obj')
selection_file = join('resources', 'mesh_selection.npy')
plt = MeshSelection(mesh_file=mesh_file)
plt.launch()
plt.save()
del plt

# Selection with an existing selection file
plt = MeshSelection(mesh_file=mesh_file,
                    selection_file=selection_file)
plt.launch()

# Selection accessors
print("Selected cells:")
print(plt.get_selected_cells_id())
print(plt.get_selected_cells_value())
print("\nSelected points:")
print(plt.get_selected_points_id())
print(plt.get_selected_points_value())
remove(selection_file)
