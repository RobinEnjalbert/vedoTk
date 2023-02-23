from os.path import join
from vedoTk import MeshSelection

# Selection from scratch
file = join('resources', 'armadillo.obj')
plt = MeshSelection(mesh_file=file)
plt.launch()
plt.save()
del plt

# Selection with an existing selection file
plt = MeshSelection(mesh_file=file,
                    selection_file=join('resources', 'mesh_selection.npy'))
plt.launch()

# Selection accessors
print("Selected cells:")
print(plt.get_selected_cells_id())
print(plt.get_selected_cells_value())
print("\nSelected points:")
print(plt.get_selected_points_id())
print(plt.get_selected_points_value())
