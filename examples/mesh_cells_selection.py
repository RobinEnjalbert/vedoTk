from os.path import join
from vedoTk import MeshCellsSelection

# Selection from scratch
mesh_file = join('resources', 'raptor.obj')
selection_file = join('resources', 'mesh_cells_selection.npy')
plt = MeshCellsSelection(mesh_file=mesh_file)
plt.launch()
plt.save(selection_file=selection_file, overwrite=True)

# Selection accessors
print("Selected cells:")
print(plt.selected_cells_id)
print(plt.selected_cells_values)
print("\nSelected points:")
print(plt.selected_points_id)
print(plt.selected_points_coord)
