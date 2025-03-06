from typing import Optional
from os import getcwd, listdir
from os.path import isfile, abspath, join, sep
from numpy import load
from vedo import Plotter, Points, Mesh, TetMesh, MeshVisual, Text2D


class FolderInspection(Plotter):

    def __init__(self, path: Optional[str] = None, extension: Optional[str] = None, debug: bool = False):
        """
        Launch a Viewer to explore the meshes file contained in a directory.
        Available file extensions are: obj, vtk, stl, npy.

        :param path: Path to the directory containing the meshes.
        :param extension: Filter the 3D objects by extension.
        """

        super().__init__(title='Folder Inspection')

        self.__actors = []
        self.__filenames = []
        self.__idx = 0

        # Filter files
        path = getcwd() if path is None else abspath(path)
        extensions = ['obj', 'vtk', 'stl', 'npy'] if extension is None else [extension.split('.')[-1]]
        files = [join(path, f) for f in sorted(listdir(path))
                 if isfile(join(path, f)) and f.split('.')[-1] in extensions]

        # Load files
        for file in files:

            if debug:
                print(f'Loading f{file}')

            # Load numpy files as point clouds
            if file.endswith('npy'):
                try:
                    actor = Points(load(file))
                except ValueError:
                    actor = None

            # Load other files as meshes
            else:
                try:
                    actor = Mesh(file)
                except AttributeError:
                    try:
                        actor = TetMesh(file)
                    except AttributeError:
                        actor = None

            # Check empty data and apply rendering style
            if actor is not None and actor.npoints > 0:
                actor.c('lightgreen')
                if isinstance(actor, MeshVisual):
                    actor.lw(0.1)
                self.__actors.append(actor)
                self.__filenames.append(file.split(sep)[-1])

        # Check that the folder is not empty
        if len(self.__actors) == 0:
            if extension is None:
                raise ValueError(f"The path {path} does not contain any mesh file.")
            raise ValueError(f"The path {path} does not contain any mesh file with extension {extension}.")

        # Add text and button to switch between files
        self.__text = Text2D(txt=self.__filenames[0], pos='bottom_middle')
        self.add(self.__text)
        self.add_button(fnc=self.__get_callback(i=-1), states=['<'], c=['w'], bc=['red3'], pos=[0.1, 0.05])
        self.add_button(fnc=self.__get_callback(i=+1), states=['>'], c=['w'], bc=['red3'], pos=[0.9, 0.05])

        # Display the first actor
        self.add(self.__actors[0])
        self.show(interactive=True, axes=4).close()

    def __get_callback(self, i: int):

        def update(obj, ename):

            # Remove the current actor
            self.remove(self.__actors[self.__idx])

            # Display the next actor
            self.__idx = (self.__idx + i) % len(self.__actors)
            self.add(self.__actors[self.__idx])

            # Change filename and render
            self.__text.text(self.__filenames[self.__idx])
            self.render(resetcam=True)

        return update
