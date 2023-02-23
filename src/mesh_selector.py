from typing import Optional
from os import listdir
from os.path import join, abspath, pardir, exists
from numpy import array, load, save, unique
from vedo import Plotter, Mesh, Text2D
from vedo.colors import get_color


class MeshSelection(Plotter):

    def __init__(self,
                 mesh_file: str,
                 selection_file: Optional[str] = None):
        """
        Implementation of a vedo.Plotter that allows selecting the cells of a mesh.

        :param mesh_file: Path to the mesh file.
        :param selection_file: Path to an existing selection file.
        """

        Plotter.__init__(self)

        # Create the mesh
        self.mesh: Mesh = Mesh(mesh_file).force_opaque().lw(0.1)
        self.mesh_dir = abspath(join(mesh_file, pardir))

        # Default selection of the mesh
        color = array(list(get_color('lightgreen')) + [1.]) * 255
        self.default_color = array([color for _ in range(self.mesh.ncells)])
        self.selected_cells = [] if selection_file is None else load(selection_file).tolist()
        self.indicator = Text2D('Nb selected cells: 0', pos='bottom-left', s=0.7)

        # Cursor to access 3D coordinates of the mesh
        self.id_cursor = -1
        self.drag_mode = False

        # Create mouse and keyboard callbacks
        self.add_callback('KeyPress', self.__callback_key_press)
        self.add_callback('MouseMove', self.__callback_mouse_move)
        self.add_callback('LeftButtonPress', self.__callback_left_click)
        self.add_callback('RightButtonPress', self.__callback_right_click)

    def launch(self, **kwargs):
        """
        Launch the Plotter. Specify Plotter.show() arguments in kwargs.
        """

        # Plotter legends
        self.render()
        instructions = "MOUSE CONTROL\n" \
                       "  Press 'd' to switch between 'click' / 'draw' modes.\n" \
                       "  Left-click to select a cell.\n" \
                       "  Right-click to unselect a cell.\n\n" \
                       "KEYBOARD CONTROL\n" \
                       "  Press 'z' to remove the last selected cell.\n" \
                       "  Press 'c' to clear the selection.\n" \
                       "  Press 's' to save the current selection."
        self.add(Text2D(txt=instructions, pos='top-left', s=0.6, bg='grey', c='white', alpha=0.9))
        self.add(self.indicator)

        # Add the mesh to the Plotter & Color the mesh with the existing selection
        self.add(self.mesh)
        self.update_mesh_colors()

        # Launch Plotter
        self.show(**kwargs).close()

    def update_mesh_colors(self, color_cursor: bool = True):
        """
        Update the mesh cells colors.
        """

        # Add the cursor to selection if a cell is flown over
        id_cells = self.selected_cells.copy()
        if self.id_cursor != -1 and color_cursor:
            id_cells += [self.id_cursor]

        # RGBA color
        selection_color = list(get_color('tomato')) + [1.]
        mesh_color = self.default_color.copy()
        mesh_color[id_cells] = array(selection_color) * 255
        self.mesh.cellcolors = mesh_color
        self.render()

    def add_cell(self, id_cell: int):
        """
        Add a new cell to the selection.
        """

        if id_cell >= 0 and id_cell not in self.selected_cells:
            self.selected_cells.append(id_cell)
            self.indicator.text(f'Nb selected cells: {len(self.selected_cells)}')
            self.update_mesh_colors()

    def remove_cell(self, id_cell: int):
        """
        Remove a cell from the selection.
        """

        if id_cell >= 0 and id_cell in self.selected_cells:
            self.selected_cells.remove(id_cell)
            self.indicator.text(f'Nb selected cells: {len(self.selected_cells)}')
            self.update_mesh_colors()

    def remove_last_cell(self):
        """
        Remove the last added cell from the selection.
        """

        if len(self.selected_cells) > 0:
            self.selected_cells.pop()
            self.indicator.text(f'Nb selected cells: {len(self.selected_cells)}')
            self.update_mesh_colors()

    def remove_all_cells(self):
        """
        Clear the whole selection.
        """

        # Clear the cell list
        self.selected_cells = []
        self.indicator.text(f'Nb selected cells: {len(self.selected_cells)}')
        self.update_mesh_colors()

    def save(self):
        """
        Save the current selection.
        """

        # Indexing file
        filename = 'mesh_selection'
        if exists(join(self.mesh_dir, f'{filename}.npy')):
            nb_file = len([file for file in listdir(self.mesh_dir) if file[:len(filename)] == filename])
            filename = f'{filename}_{nb_file}'

        # Save selection
        save(join(self.mesh_dir, filename), array(self.selected_cells, dtype=int))

    def get_selected_cells_id(self):
        """
        Get the ids of the selected cells.
        """

        return array(sorted(self.selected_cells))

    def get_selected_cells_value(self):
        """
        Get the values of the selected cells.
        """

        return array(self.mesh.cells())[self.get_selected_cells_id()]

    def get_selected_points_id(self):
        """
        Get the ids of the selected points.
        """

        return unique(array(self.get_selected_cells_value()).flatten())

    def get_selected_points_value(self):
        """
        Get the values of the selected points.
        """

        return array(self.mesh.points())[self.get_selected_points_id()]

    def __callback_key_press(self, event):
        """
        KeyPressEvent callback.
        """

        # If 's' pressed, switch drag_mode flag
        if event.keyPressed == 'd':
            self.drag_mode = not self.drag_mode

        # If 'z' pressed, remove the last selected cell
        elif event.keyPressed == 'z' and len(self.selected_cells) > 0:
            self.remove_last_cell()

        # If 'c' pressed, clear the selection
        elif event.keyPressed == 'c':
            self.remove_all_cells()

        # If 's' pressed, save the selection
        elif event.keyPressed == 's':
            self.save()

    def __callback_mouse_move(self, event):
        """
        MouseMoveEvent callback.
        """

        # Mouse is on the object : color the current cell
        if event.actor is not None:
            self.id_cursor = self.mesh.closest_point(event.picked3d, return_cell_id=True)
            if self.drag_mode:
                self.add_cell(id_cell=self.id_cursor)
            self.update_mesh_colors()

        # Mouse is not on the object: uncolor the previous cell
        else:
            if self.id_cursor != -1:
                self.id_cursor = -1
                self.update_mesh_colors()

    def __callback_left_click(self, event):
        """
        LeftButtonPressEvent callback.
        """

        # Mouse is on the object: color the clicked cell
        if event.actor is not None:
            self.id_cursor = self.mesh.closest_point(event.picked3d, return_cell_id=True)
            self.add_cell(id_cell=self.id_cursor)
            self.update_mesh_colors()

    def __callback_right_click(self, event):
        """
        RightButtonPressEvent callback.
        """

        # Mouse is on the object: uncolor the clicked cell
        if event.actor is not None:
            self.id_cursor = self.mesh.closest_point(event.picked3d, return_cell_id=True)
            self.remove_cell(id_cell=self.id_cursor)
            self.update_mesh_colors(color_cursor=False)
