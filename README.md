# Vedo Toolkit

A Python toolbox based on [vedo](https://github.com/marcomusy/vedo) for 3D objects visualization and manipulation.

## Installation

```bash
# Option 1 (USERS): install with pip
$ pip install git+https://github.com/RobinEnjalbert/vedoTk.git

# Option 2 (DEVS): install as editable
$ git clone https://github.com/RobinEnjalbert/vedoTk.git
$ cd vedoTk
$ pip install -e .
```


## Usage

### Folder inspection

This tool allows you to easily explore the mesh files contents in a repository. 
The window displays the loaded meshes one by one, and you can switch between files with buttons.
```python
from vedoTk import FolderInspection
FolderInspection('examples/resources')
```
![folder_inspection](examples/img/folder_inspection.png)
See `examples/folder_inspection.py`.


### Mesh selection

This tool allows you to easily select cells or points from a surface mesh with the mouse.
The indices and positions can be saved in a `.npy` file.
```python
from vedoTk import MeshCellsSelection
MeshCellsSelection('examples/resources/raptor.obj').launch()
```
![mesh_selection](examples/img/mesh_selection.png)
See `examples/mesh_points_selection.py` and `examples/mesh_cells_selection.py`.


### Visible points

This tool allows you to extract the visible points of a Mesh depending on the virtual camera parameters.
This tool should be used when the extraction is done offscreen (otherwise, use `vedo.visible_pints`).
```python
from vedo import Mesh
from vedoTk import VisiblePoints
VisiblePoints(Mesh('examples/resources/armadillo.obj')).extract()
```
![visible_points](examples/img/visible_points.png)
See `examples/visible_points.py`.