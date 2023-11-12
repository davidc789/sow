""" Investigation library with many useful tools.

:author: David Chen <chdc@student.unimelb.edu.au>
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any, Callable, Literal, TypeVar, Generic

from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
import statsmodels.api as sm

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from collections import deque, Counter

T = TypeVar("T")


@dataclass
class SimulationContext(object):
    """ A generic simulation context. """
    model: "Model"     # Reference to the model


@dataclass
class SimulationStartContext(SimulationContext):
    """ Context to the start of the simulation. """
    t_lim: int         # The end time of the simulation.
    hard_stop: bool    # Whether to stop simulations regardless of topple status.


@dataclass
class SimulationStepContext(SimulationContext):
    """ Context to the current step of simulation. """
    vertex: int        # The index of the cell selected.
    is_topple: bool    # Whether the step is considered a topple step.


# Type alias for the subsequent family of functions.
ContextTransformerType = Callable[[SimulationContext], T]


class ContextTransformer(object):
    """ A static class holding context-transforming utilities.

    All the functions receive a context (maybe a specific type) and outputs
    some results that can be used in the listeners.
    """
    @staticmethod
    def visualise_graph(context: SimulationContext):
        """ Visualise the simulation as a graph.

        :param context: The simulation context.
        :return: Matplotlib figure and axes.
        """
        return visualise_graph(
            context.model.original_graph, context.model.height,
            max_height=context.model.topple_limit)

    @staticmethod
    def mean_height(context: SimulationContext):
        """ Computes the mean height of the graph sand pile.

        :param context: The simulation context.
        :return: The mean height.
        """
        return np.mean(context.model.height)

    @staticmethod
    def topple_occurrence(context: SimulationContext):
        """ Tracks when the sand pile topples.

        :param context: The simulation context.
        :return: The topple frequency.
        """
        if isinstance(context, SimulationStepContext):
            return context.is_topple
        else:
            return np.nan

    @staticmethod
    def sand_loss(context: SimulationContext):
        """ Gives the number of sand lost due to toppling over the boundary.

        :param context: The simulation context.
        :return: Number of sand lost during toppling.
        """
        if isinstance(context, SimulationStepContext):
            if context.vertex in context.model.boundary_vertices:
                return context.model.boundary_vertices[context.vertex]
            else:
                return 0
        else:
            return np.nan

    @staticmethod
    def drop_location(context: SimulationContext):
        """ Gives the location of sand drop.

        :param context: The simulation context.
        :return: Vertex index at which sand is dropped onto.
        """
        if isinstance(context, SimulationStepContext):
            return context.vertex
        else:
            return np.nan


class SimulationListener(ABC):
    """ The base class for simulation listener callback objects.

    Although this class is instantiable, it captures nothing from the simulation
    as a listener, so you should only use this class as a parent class to your
    custom listener implementations.

    Ideally, one should design the listeners such that they can be reused across
    multiple models. This is made possible by the simulation start and
    simulation end method, which should be used to control the life cycle of the
    listener.

    For memory efficiency, it is also recommended to remove references to large
    temporary variables once they are no longer needed, unless they need to be
    accessed by the user after the simulation life cycle.
    """
    def onSimulationStart(self, context: SimulationStartContext):
        """ A call-back triggered when the simulation starts.

        :param context: The simulation context.
        """
        pass

    def onSimulationEnd(self, context: SimulationContext):
        """ A call-back triggered when the simulation ends.

        :param context: The simulation context.
        """
        pass

    def onSimulationStep(self, context: SimulationStepContext):
        """ A call-back triggered when a step of simulation passed.

        :param context: The simulation context.
        """
        pass


class SimulationManager(SimulationListener, ABC):
    """ The base class for simulation manager callback objects. """
    def chooseSimulationStep(self, context: SimulationContext) -> int:
        """ A call-back used to determine the next simulation step.

        :param context: The simulation context.
        :return: Index of the next vertex to drop sand on. This is ignored when
        the sandpile is toppling.
        """
        pass


class NullListener(SimulationListener):
    """ A listener that does nothing. Useful for debug and testing. """
    def __init__(self):
        pass


class NullManager(SimulationManager):
    """ A manager that does nothing. Useful for debug and testing. """
    def __init__(self):
        pass


class ListenerGroup(SimulationListener):
    """ Group together smaller listeners to create a larger listener.

    Note that the group is a listener itself, so it is possible to create nested
    listener groups, although a flat structure is more computationally
    efficient.
    """
    listeners: list[SimulationListener]

    def __init__(self, listeners: list[SimulationListener]):
        self.listeners = listeners

    def onSimulationStart(self, context: SimulationStartContext):
        for listener in self.listeners:
            listener.onSimulationStart(context)

    def onSimulationEnd(self, context: SimulationContext):
        for listener in self.listeners:
            listener.onSimulationEnd(context)

    def onSimulationStep(self, context: SimulationStepContext):
        for listener in self.listeners:
            listener.onSimulationStep(context)


class ConstantDropper(SimulationManager):
    """ A simulation manager that drops sand at the same spot repeatedly. """
    _drop_index: int

    def __init__(self, drop_index: int):
        """ Constructs a constant sand dropper.

        :param drop_index: The vertex index to drop sand on in any run.
        """
        self._drop_index = drop_index

    @property
    def drop_index(self):
        """ The vertex index to drop sand onto. """
        return self._drop_index

    def chooseSimulationStep(self, context: SimulationContext) -> int:
        return self._drop_index


class RandomDropper(SimulationManager):
    """ A simulation manager that drops sand at random. """
    _all_u: Optional[np.ndarray[int]]

    def __init__(self):
        pass

    def chooseSimulationStep(self, context: SimulationContext) -> int:
        return self._all_u[context.model.t]

    def onSimulationStart(self, context: SimulationStartContext):
        # Prepares a sequence of random numbers in advance to leverage the
        # power of vectorisation.
        self._all_u = np.random.randint(
            0, context.model.original_graph.vcount(), context.t_lim)

    def onSimulationEnd(self, context: SimulationContext):
        # Clean up the reference to the cached random integers.
        self._all_u = None


class SimulationHistoryRecorder(SimulationListener):
    """ Tracks the history of simulations. """
    _records: list[tuple[int, list[int]]]
    _frequency: int
    _store_init: bool

    def __init__(self, frequency: int = 1, store_init: bool = True):
        """ Records simulation history.

        :param frequency: The frequency at which simulations are stored.
        :param store_init: Whether to store the initial heights.
        """
        super().__init__()
        self._frequency = frequency
        self._store_init = store_init

    @property
    def records(self):
        """ Gets the list of stored records. """
        return self._records

    def record_history(self, context: SimulationContext):
        """ Record history using the given context.

        If one wish to change the records stored by the history recorder,
        inherit and overwrite this method.

        :param context: The simulation context.
        """
        return context.model.t, context.model.height.copy()

    def onSimulationStart(self, context: SimulationContext):
        # Initialise the records and stores the initial grid if required.
        # A copy is necessary.
        self._records = []

        if self._store_init:
            self._records.append(self.record_history(context))

    def onSimulationStep(self, context: SimulationStepContext):
        # Make a snapshot if required.
        if context.model.t % self._frequency == 0:
            self._records.append(self.record_history(context))


class ImageMaker(SimulationListener):
    """ Makes a series of images from the simulation. """
    # Type aliases.
    Visualiser = ContextTransformerType[tuple[Figure, Axes]]

    # Final variables fixed after initialisation.
    out_dir: Path | None = None
    save_only: bool
    ext: str
    visualiser: Visualiser
    subfig: bool

    # Internal states.
    images: list[tuple[Figure, Axes]] | None = None

    def __init__(self, out_dir: str | None = None, save_only: bool = False,
                 ext: str = "jpg", visualiser: Visualiser | None = None,
                 subfig: bool = False):
        """ Creates an image maker.

        :param out_dir: Directory for output. If None the images will still
        be tracked but no files will be saved.
        :param save_only: Whether to not store the history of files. This
        requires out_dir to be specified
        :param ext: The filename extension for the images.
        :param visualiser: A callback that generates the frames.
        :param subfig: Whether to generate all images in one figure.
        """
        self.save_only = save_only
        self.ext = ext
        self.subfig = subfig

        if out_dir is not None:
            self.out_dir = Path(out_dir)

        # Apply the default visualiser if unspecified.
        if visualiser is None:
            self.visualiser = ContextTransformer.visualise_graph
        else:
            self.visualiser = visualiser

    def onSimulationStart(self, context: SimulationStartContext):
        # Initialise internal states.
        self.images = []

    def onSimulationEnd(self, context: SimulationContext):
        if self.save_only:
            self.images = None

    def onSimulationStep(self, context: SimulationStepContext):
        # Create the plot.
        fig, ax = self.visualiser(context)
        if self.save_only:
            fig.savefig(self.out_dir / f"{context.model.t}.{self.ext}")
        else:
            self.images.append((fig, ax))


class MovieMaker(SimulationListener):
    """ Makes a movie from the simulation.

    It is recommended to set hard_stop = True in the model when using this
    listener because it relies on a precise t_lim to run correctly.
    """
    # Final variables fixed after initialisation.
    fps: int
    max_frame: int
    duration: float
    out_dir: str | None = None

    # State variables tracked during simulations.
    animation: Any = None
    _frames: list[np.ndarray] | None = None
    _frame_number: int | None = None
    _max_height: int | None = None
    _t_lim: int | None = None
    _fig: Figure | None = None
    _ax: Axes | None = None

    def __init__(self, fps: int = 30, max_frame: int | None = None,
                 duration: float | None = None, out_dir: str | None = None):
        """ Constructs a simulation movie creator.

        For simplicity, this listener assumes a fixed fps while exactly only one
        of max_frame and duration must be specified.
        An error will be raised if none or both are supplied.
        Note that the generated video may have a duration slightly different to
        the duration specified, since rounding is applied when choosing frames.

        :param fps: Frames per second. Typically, they are multiples of 30 but
        this is not a requirement.
        :param max_frame: The maximum number of frames.
        :param duration: Duration of the video, in seconds.
        :param out_dir: Directory for video output. The extension name also
        alters the format of the saved file. If None the animation will still
        be returned but no files will be saved.
        """
        self.fps = fps
        self.out_dir = out_dir

        # Check the inputs.
        if max_frame is None and duration is None:
            raise ValueError("At least one of max_frame and duration needs to "
                             "be specified")
        elif max_frame is not None and duration is not None:
            raise ValueError("At most one of max_frame and duration can be "
                             "specified")

        # Compute other quantities using the values available.
        if max_frame is None:
            self.duration = duration
            self.max_frame = round(fps * duration)
        else:
            self.duration = max_frame / fps
            self.max_frame = max_frame

    def make_frame(self, context: SimulationStepContext, fig: Figure, ax: Axes):
        """ Makes a frame based on the current context.

        Override this method to customise the video generated.

        :param context: Simulation context.
        :param fig: Matplotlib figure.
        :param ax: Matplotlib axes.
        """
        # Clear the axes before adding the next plot.
        ax.clear()
        visualise_graph(context.model.original_graph, context.model.height,
                        max_height=context.model.topple_limit, fig=fig, ax=ax)
        return mplfig_to_npimage(fig)

    def onSimulationStart(self, context: SimulationStartContext):
        # Input checks.
        if not context.hard_stop:
            warnings.warn("hard_stop is disabled and the frames may not have "
                          "been calculated correctly")
        if context.t_lim < self.max_frame:
            warnings.warn("A small t_lim is detected and the frames are "
                          "insufficient. The same frame might be duplicated.")

        # Initialise internal states.
        self._frame_number = 0
        self._frames = []
        self._max_height = context.model.topple_limit
        self._t_lim = context.t_lim
        self._fig, self._ax = plt.subplots()

    def onSimulationEnd(self, context: SimulationContext):
        # State checks.
        if (self._frame_number != self.max_frame
                or len(self._frames) != self._frame_number):
            raise RuntimeWarning("Invalid internal state.")

        def _make_frame(t: float):
            """ Internal helper function to make frames.

            :param t: The current video time.
            """
            return self._frames[round(t / self.duration) * self.max_frame]

        # Create the animation.
        animation = VideoClip(make_frame=_make_frame, duration=self.duration)
        # autoplay=True

        # Reset the states.
        self._frames = None
        self._frame_number = None
        self._fig = None
        self._ax = None

        if self.out_dir is not None:
            animation.write_videofile("./out/video.mp4", fps=20, loop=True)

        self.animation = animation

    def onSimulationStep(self, context: SimulationStepContext):
        # Logic to determine whether the current frame should be included.
        # This is a significant frame if the time of simulation
        t = context.model.t
        while t / self._t_lim >= self._frame_number / self.max_frame:
            frame = self.make_frame(context, self._fig, self._ax)
            self._frames.append(frame)
            self._frame_number += 1


class StatisticsCollector(SimulationListener, Generic[T]):
    """ A generic statistics collector. """
    calc: ContextTransformerType[T]
    store_history: bool
    value: float | None = None
    value_history: list[float] | None = None

    def __init__(self, calc: ContextTransformerType[T],
                 store_history: bool = True):
        """ Constructs a statistics collector.

        :param calc: Determine how the statistic is calculated.
        :param store_history: Whether to store history.
        """
        self.calc = calc
        self.store_history = store_history

    def onSimulationStart(self, context: SimulationStartContext):
        # Initialise the means.
        self.value = None
        if self.store_history:
            self.value_history = []

    def onSimulationStep(self, context: SimulationStepContext):
        # The mean height is simply the mean of all heights on the grid.
        if self.store_history:
            self.value = self.calc(context)
            self.value_history.append(self.value)

    def onSimulationEnd(self, context: SimulationContext):
        # Store the final grid mean height.
        self.value = self.calc(context)


@dataclass
class Model(object):
    """ The main model.

    Here, the boundary_vertices dictionary stores the number of sinks a vertex
    is connected to. This is useful because for a grid model, the corners are
    connected to two sinks while other edges are only connected to one sink.
    """
    # Fields that are final once initialised.
    graph: ig.Graph
    boundary_vertices: dict[int, int] = field(default_factory=lambda: {})
    topple_limit: int = 4
    t_lim: int = 1_000_000
    hard_stop: bool = False

    # Advanced fields for customising the simulation and subscribing to the
    # simulation events.
    manager: SimulationManager = RandomDropper()
    listeners: list[SimulationListener] = ()

    # Fields that are indirectly initialised.
    original_graph: ig.Graph = field(init=False)
    sink_indices: list[int] = field(init=False, default_factory=lambda: [])

    # The parameters that changes as the model evolves.
    t: int = field(default=0, init=False)
    height: list[int] = field(init=False)

    def __post_init__(self):
        """ A special dataclass method that is called after __init__. """
        # Initialise the height vector and standardise the list fields.
        self.height = [0 for _ in range(self.graph.vcount())]
        self.listeners = list(self.listeners)

        # Makes a copy of the graph.
        self.original_graph = self.graph
        self.graph = self.graph.copy()

        # Adds the sink vertices.
        sink_count = max(self.boundary_vertices.values())
        self.sink_indices = [self.graph.vcount() + i for i in range(sink_count)]
        self.graph.add_vertices(sink_count)

        # Joins the sink nodes and the boundary vertices, using their
        # corresponding sink count.
        for u, count in self.boundary_vertices.items():
            self.graph.add_edges([(u, v) for v in self.sink_indices[:count]])

    def add_listener(self,
                     listener: SimulationListener | list[SimulationListener]):
        """ Adds listeners to the current simulation.

        :param listener: A simulation listener or a list of listeners.
        :return: Reference to itself for chaining.
        """
        if isinstance(listener, SimulationListener):
            self.listeners.append(listener)
        else:
            for listener in listener:
                self.listeners.append(listener)
        return self

    def simulate(self, t_lim: int = 10_000_000, hard_stop: bool = False):
        """ Performs the simulation.

        :param t_lim: The maximum time step to run this simulation.
        :param hard_stop: Whether the time limit is strict. If True then the simulation
        will stop exactly at t_lim and unstable sand cells may exist; otherwise,
        the simulation stops after the current toppling ends.
        """
        # Invoke the simulation start callbacks.
        start_context = SimulationStartContext(self, t_lim, hard_stop)
        self.manager.onSimulationStart(start_context)
        for listener in self.listeners:
            listener.onSimulationStart(start_context)

        while self.t < t_lim:
            # Fetch the next vertex from the manager and add sand to it.
            u = self.manager.chooseSimulationStep(SimulationContext(self))
            self.height[u] += 1
            self.t += 1

            # The step is concluded, so make the step callbacks.
            step_context = SimulationStepContext(self, u, False)
            self.manager.onSimulationStep(step_context)
            for listener in self.listeners:
                listener.onSimulationStep(step_context)

            # If toppling is triggered, perform them in a sub-routine.
            if self.height[u] >= self.topple_limit:
                self._topple(u)

        # The simulation is concluded, so make the end callbacks.
        end_context = SimulationContext(self)
        self.manager.onSimulationEnd(end_context)
        for listener in self.listeners:
            listener.onSimulationEnd(end_context)

    def _topple(self, original_u: int):
        """ Performs topple on a given vertex until completion.

        :param original_u: The starting point of toppling.
        """
        queue = deque([original_u])

        while len(queue) > 0:
            if self.t == self.t_lim and self.hard_stop:
                print("Oh no, the toppling is abruptly stopped due to "
                      "'hard_stop' being set to True. If you don't mind "
                      "waiting a couple more milliseconds, set it to False.")
                return

            # Perform grid topple and spread the sand on cell (i, j).
            u = queue.popleft()
            self.height[u] -= 4
            self.t += 1

            # If this is not enough, adds the cell back into the queue.
            if self.height[u] >= self.topple_limit:
                print("Wha? The topology must be weird. Consider lifting the "
                      "topple limit.")
                queue.append(u)

            # Gets the neighbour coordinate.
            for v in self.graph.neighbors(u):
                # If the new cell is within the boundary, give it sand; if that
                # makes it full, adds it into the topple search stack;
                # otherwise, let it drop into the void.
                if v not in self.sink_indices:
                    self.height[v] += 1

                    # If the new cell is full, adds it into the queue and
                    # perform toppling on it. Avoid repeated queuing by
                    # enforcing equality.
                    if self.height[v] == self.topple_limit:
                        queue.append(v)

            # The step is concluded, so make the callbacks.
            step_context = SimulationStepContext(self, u, True)
            self.manager.onSimulationStep(step_context)
            for listener in self.listeners:
                listener.onSimulationStep(step_context)

    def visualise_graph(self, *args, **kwargs):
        """ A shortcut to the visualise function. """
        return visualise_graph(self.original_graph, self.height, *args, **kwargs)

    def visualise_grid(self, n: int, m: int, *args, **kwargs):
        """ A shortcut to the visualise_as_grid function. """
        return visualise_grid(n, m, self.height, *args, **kwargs)


def make_grid_graph(m: int, n: int):
    """ Makes a graph with a grid-like topology.

    :param m: Number of rows in the grid.
    :param n: Number of columns in the grid.
    :return: The resulting graph along with the set of boundary vertices.
    """
    # Construct the edges in the grid graph. We count row-wise, that is, from
    # left to right in the first row, and proceed onto the next row once the row
    # is exhausted.
    horizontals = [(i * n + j, i * n + (j + 1))
                   for i in range(m) for j in range(n - 1)]
    verticals = [(i * n + j, (i + 1) * n + j)
                 for i in range(m - 1) for j in range(n)]
    edges = horizontals + verticals

    # Construct the graph. The grid should have a boundary multiplicity of 2 in
    # the corners and 1 on the other parts of edges.
    graph = ig.Graph(edges)
    boundary_vertices = {
        i * n + j: 1 for i in range(1, m - 1) for j in range(1, n - 1)
    } | {
        i * n + j: 2 for i in [0, m - 1] for j in [0, n - 1]
    }
    return graph, boundary_vertices


def make_diagonal_grid_graph(m: int, n: int):
    """ Makes a graph with a grid-like topology, including the four diagonals.

    :param m: Number of rows in the grid.
    :param n: Number of columns in the grid.
    :return: The resulting graph along with the set of boundary vertices.
    """
    # Construct the edges in the grid graph. We count row-wise, that is, from
    # left to right in the first row, and proceed onto the next row once the row
    # is exhausted.
    horizontals = [(i * n + j, i * n + (j + 1))
                   for i in range(m) for j in range(n - 1)]
    verticals = [(i * n + j, (i + 1) * n + j)
                 for i in range(m - 1) for j in range(n)]
    diagonal_1 = [(i * n + j, (i + 1) * n + (j + 1)) for i in range(m)
                  for j in range(n) if 0 <= (i + 1) * n + (j + 1) < m * n]
    diagonal_2 = [(i * n + j, (i + 1) * n + (j - 1)) for i in range(m)
                  for j in range(n) if 0 <= (i + 1) * n + (j - 1) < m * n]
    edges = horizontals + verticals + diagonal_1 + diagonal_2

    # Construct the graph. The grid should have a boundary multiplicity of 5 in
    # the corners and 3 in the other parts of edges.
    graph = ig.Graph(edges)
    boundary_vertices = {
        i * n + j: 5 for i in range(1, m - 1) for j in range(1, n - 1)
    } | {
        i * n + j: 3 for i in [0, m - 1] for j in [0, n - 1]
    }
    return graph, boundary_vertices


def make_cylinder_graph(m: int, n: int):
    """ Makes a grid with the left and right boundaries connected.

    The resulting graph is effectively isomorphic to a cylinder with openings on
    the top and bottom.

    :param m: Number of rows in the grid.
    :param n: Number of columns in the grid.
    :return: The resulting graph along with the set of boundary vertices.
    """
    # Construct the edges in the grid graph. We count row-wise, that is, from
    # left to right in the first row, and proceed onto the next row once the row
    # is exhausted.
    # The horizontal connections should wrap around.
    horizontals = [(i * n + j, i * n + (j + 1) % n)
                   for i in range(m) for j in range(n)]
    verticals = [(i * n + j, (i + 1) * n + j)
                 for i in range(m - 1) for j in range(n)]
    edges = horizontals + verticals

    # Construct the graph. The grid should have a boundary multiplicity of 1 on
    # the top and bottom edges.
    graph = ig.Graph(edges)
    boundary_vertices = {
        i * n + j: 1 for i in [0, m - 1] for j in range(n - 1)
    }
    return graph, boundary_vertices


def make_nary_tree(n: int, d: int):
    """ Makes a complete n-ary tree with depth d.

    :param n: The offspring multiplicity of the tree.
    :param d: The depth of the tree.
    :return: The resulting graph along with the set of boundary vertices.
    """
    internal_node_count = (n ** d - 1) // (n - 1)
    total_node_count = (n ** (d + 1) - 1) // (n - 1)
    edges = [
        (u, n * u + 1 + i) for i in range(n) for u in range(internal_node_count)
    ]

    # Construct the graph. The graph should have a boundary multiplicity of 3.
    graph = ig.Graph(edges)
    boundary_vertices = {
        u: 3 for u in range(internal_node_count, total_node_count)
    }
    return graph, boundary_vertices


def make_interesting_tree(n: int, d: int):
    """ Makes an interesting graph.

    :param n: Number of
    :param d: Depth of the sub-trees.
    :return: The resulting graph along with the set of boundary vertices.
    """
    nary_tree, nary_tree_boundary_vertices = make_nary_tree(n, d)
    nary_tree_edges = nary_tree.get_edgelist()
    node_count = nary_tree.vcount()
    edges = ([(u + w * node_count + 1, v + w * node_count + 1)
              for u, v in nary_tree_edges for w in range(n + 1)]
             + [(0, w * node_count + 1) for w in range(n + 1)])

    # Construct the graph. The graph should have a boundary multiplicity of 3.
    graph = ig.Graph(edges)
    boundary_vertices = {
        u + w * node_count + 1: nary_tree_boundary_vertices[u]
        for u in nary_tree_boundary_vertices for w in range(n + 1)
    }
    return graph, boundary_vertices


def make_punctured_donut_graph(m: int, n: int):
    """ Makes a grid with the left and right, top and bottom connected.

    The resulting graph is effectively isomorphic to a donut with an extra hole.
    Note that to make the central vertex well-defined, both m and n should be
    odd values.

    :param m: Number of rows in the grid.
    :param n: Number of columns in the grid.
    :return: The resulting graph along with the set of boundary vertices.
    """
    if m % 2 == 0 or n % 2 == 0:
        warnings.warn("Both m and n should be odd, or you may see the "
                      "punctured hole not in the center.")

    # Construct the edges in the grid graph. We count row-wise, that is, from
    # left to right in the first row, and proceed onto the next row once the row
    # is exhausted.
    # Both horizontal and vertical connections should wrap around.
    horizontals = [(i * n + j, i * n + (j + 1) % n)
                   for i in range(m) for j in range(n)]
    verticals = [(i * n + j, ((i + 1) % m) * n + j)
                 for i in range(m) for j in range(n)]
    edges = horizontals + verticals

    # Construct the graph. Remove a center vertex and make its neighbours
    # boundaries.
    graph = ig.Graph(edges)
    center_vertex = (n * m) // 2
    graph.delete_vertices(center_vertex)
    boundary_vertices = {
        center_vertex + i * n + j: 1
        for i, j in [(-1, 0), (1, 0), (0, -1), (0, 1)]
    }
    return graph, boundary_vertices


def visualise_graph(
        graph: ig.Graph, height: list[int], max_height=None,
        fig: Figure = None, ax: Axes = None):
    """ Produce an image of the grid.

    :param graph: The graph of the model.
    :param height: The height vector.
    :param max_height: Maximum height of the grid. By default, it will be
    inferred from height.
    :param fig: Matplotlib figure. If not specified, a new (figure, axis) pair
    will be created.
    :param ax: Matplotlib axis. If not specified, a new (figure, axis) pair
    will be created.
    :return: Figure, axes pair with the graph plotted.
    """
    if max_height is None:
        max_height = max(height)

    # Compute the vertex colors.
    palette = ig.drawing.colors.GradientPalette("blue", "red", n=max_height + 1)
    vertex_color = palette.get_many(height)

    # Plot!
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    ig.plot(graph, target=ax, vertex_color=vertex_color)

    # Return the figure and user can do whatever they want with it.
    return fig, ax


def visualise_grid(
        m: int, n: int, height: list[int], max_height: int = None,
        fig: Figure = None, ax: Axes = None, xticks=None, yticks=None,
        labels=None, title=None):
    """ Visualise the graph model as a grid.

    :param m: Number of rows.
    :param n: Number of columns.
    :param height: The height vector.
    :param max_height: Maximum height of the grid. By default, it will be
    inferred from height.
    :param fig: Matplotlib figure. If not specified, a new (figure, axis) pair
    will be created.
    :param ax: Matplotlib axis. If not specified, a new (figure, axis) pair
    will be created.
    :param xticks: Labels for the x-axis.
    :param yticks: Labels for the y-axis.
    :param labels: Grid cell annotations.
    :param title: Plot title.
    :return: Figure, axes pair with the graph plotted.
    """
    # Gridify the graph.
    grid = [[height[i * m + j] for j in range(n)] for i in range(m)]

    # Create the awesome plot.
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    ax.imshow(grid, cmap="Greys", vmin=0, vmax=max_height, interpolation="none")

    # Show ticks and label them with the respective entries.
    if xticks is not None:
        if xticks == "auto":
            xticks = [str(x) for x in range(n)]
        ax.set_xticks(np.arange(len(xticks)), labels=xticks)
    if yticks is not None:
        if yticks == "auto":
            yticks = [str(x) for x in range(m)]
        ax.set_yticks(np.arange(len(yticks)), labels=yticks)

    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if labels is not None:
        if labels == "auto":
            labels = [[str(x) for x in row] for row in grid]

        for i in range(m):
            for j in range(n):
                ax.text(j, i, labels[i][j], ha="center", va="center")

    # Set the title.
    ax.set_title(title)
    fig.tight_layout()

    # Return the figure and user can do whatever they want with it.
    return fig, ax


def avalanche_statistics(topple_occurrence: list[bool],
                         topple_loss: list[int], drop_location: list[int]):
    """ Computes a range of avalanche statistics.

    The statistics computed are the duration, loss and area of the avalanche

    :param topple_occurrence: Flags for when topple occurred.
    :param topple_loss: The loss when topple occurred.
    :param drop_location: Locations of toppling.
    :return: The computed statistics.
    """
    t = 0
    loss = []
    duration = []
    area = []

    while t < len(topple_occurrence):
        current_loss = 0
        current_duration = 0
        vertex_covered = set()

        # Advance to the first point toppling start to occur.
        while (t < len(topple_occurrence) and
               (topple_occurrence[t] == np.nan or not topple_occurrence[t])):
            t += 1

        # Sum all the loss for the toppling entries.
        while (t < len(topple_occurrence) and
               (topple_occurrence[t] != np.nan and topple_occurrence[t])):
            current_loss += topple_loss[t]
            current_duration += 1
            vertex_covered.add(drop_location[t])
            t += 1

        if t < len(topple_occurrence):
            loss.append(current_loss)
            duration.append(current_duration)
            area.append(len(vertex_covered))

    return duration, loss, area


if __name__ == "__main__":
    # Testing.
    graph_, boundary_vertices_ = make_grid_graph(10, 10)

    model_ = Model(
        graph=graph_,
        boundary_vertices=boundary_vertices_
    )
    model_.simulate()
    fig_, ax_ = model_.visualise_graph()
    fig_.savefig("./out/1.jpg")
