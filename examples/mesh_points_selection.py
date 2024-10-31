from os.path import join
from vedoTk import MeshPointsSelection

# Selection from scratch
mesh_file = join('resources', 'raptor.obj')
selection_file = join('resources', 'mesh_points_selection.npy')
plt = MeshPointsSelection(mesh_file=mesh_file)
plt.launch()
plt.save(selection_file=selection_file, overwrite=True)

# Selection accessors
print("\nSelected points:")
print(plt.selected_points_id)
print(plt.selected_points_coord)
