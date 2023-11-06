""" Investigation library with many useful tools. """

from dataclasses import dataclass, field
from typing import Optional

import matplotlib.figure
import numpy as np
import matplotlib.pyplot as plt
import igraph as ig

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
from collections import deque


@dataclass
class SimulationContext(object):
    """ A generic simulation context. """
    model: "Model"    # Reference to the model


@dataclass
class SimulationStartContext(SimulationContext):
    """ Context to the start of the simulation. """
    t_lim: int        # The end time of the simulation.
    hard_stop: bool   # Whether to stop simulations regardless of topple status.


@dataclass
class SimulationStepContext(SimulationContext):
    """ Context to the current step of simulation. """
    i: int            # The index of the cell selected.
    is_topple: bool   # Whether the step is considered a topple step.


class SimulationListener(object):
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


class SimulationManager(SimulationListener):
    """ The base class for simulation manager callback objects. """
    def chooseSimulationStep(self, context: SimulationContext) -> int:
        """ A call-back used to determine the next simulation step.

        :param context: The simulation context.
        :return: Index of the next vertex to drop sand on. This is ignored when
        the sandpile is toppling.
        """
        pass


class ConstantDropper(SimulationManager):
    """ A simulation manager that drops sand at the same spot repeatedly. """
    u: int

    def __init__(self, u: int):
        """ Constructs a constant sand dropper.

        :param u: The vertex index to drop sand on in any run.
        """
        self.u = u

    def chooseSimulationStep(self, context: SimulationContext) -> int:
        return self.u


class RandomDropper(SimulationManager):
    """ A simulation manager that drops sand at random. """
    all_u: Optional[np.ndarray[int]]

    def chooseSimulationStep(self, context: SimulationContext) -> int:
        return self.all_u[context.model.t]

    def onSimulationStart(self, context: SimulationStartContext):
        # Prepares a sequence of random numbers in advance to leverage the
        # power of vectorisation.
        self.all_u = np.random.randint(
            0, context.model.graph.vcount() - 1, context.t_lim)

    def onSimulationEnd(self, context: SimulationContext):
        # Clean up the reference to the cached random integers.
        self.all_u = None


class SimulationHistoryRecorder(SimulationListener):
    """ Tracks the history of simulations. """
    records: list[tuple[int, list[int]]]
    frequency: int
    store_init: bool

    def __init__(self, frequency: int = 1, store_init: bool = True):
        """ Records simulation history.

        :param frequency: The frequency at which simulations are stored.
        :param store_init: Whether to store the initial heights.
        """
        super().__init__()
        self.frequency = frequency
        self.store_init = store_init

    def onSimulationStart(self, context: SimulationStartContext):
        # Initialise the records and stores the initial grid if required.
        # A copy is necessary.
        self.records = []

        if self.store_init:
            self.records.append((context.model.t, context.model.height.copy()))

    def onSimulationStep(self, context: SimulationStepContext):
        # Make a snapshot if required.
        if context.model.t % self.frequency == 0:
            self.records.append((context.model.t, context.model.height.copy()))


@dataclass
class Model(object):
    """ The main model. """
    # Fields that are final once initialised.
    graph: ig.Graph
    boundary_vertices: list[int] = ()
    topple_limit: int = 4
    t_lim: int = 1_000_000
    hard_stop: bool = False

    # Advanced fields for customising the simulation and subscribing to the
    # simulation events.
    manager: SimulationManager = RandomDropper()
    listeners: list[SimulationListener] = ()

    # Fields that are indirectly initialised.
    original_graph: ig.Graph = field(init=False)
    sink_index: int = field(init=False)

    # The parameters that changes as the model evolves.
    t: int = field(default=0, init=False)
    height: list[int] = field(init=False)

    def __post_init__(self):
        """ A special dataclass method that is called after __init__. """
        # Initialise the height vector and standardise the list fields.
        self.height = [0 for _ in range(self.graph.vcount())]
        self.boundary_vertices = list(self.boundary_vertices)
        self.listeners = list(self.listeners)

        # Makes a copy of the graph before adding the sink.\
        self.original_graph = self.graph
        self.graph = self.graph.copy()
        self.sink_index = self.graph.vcount()
        self.graph.add_vertices(1)
        self.graph.add_edges([(u, self.sink_index)
                              for u in self.boundary_vertices])

    def add_listener(self, listener: SimulationListener | list[SimulationListener]):
        """ Adds listeners to the current simulation.

        :param listener: A simulation listener or a list of listeners.
        :return: Reference to itself for chaining.
        """
        if isinstance(listener, SimulationListener):
            self.listeners.append(listener)
        else:
            for l in listener:
                self.listeners.append(l)
        return self

    def simulate(self, t_lim: int = 10_000_000, store_freq: int = 10_000_000,
                 hard_stop: bool = False):
        """ Performs the simulation.

        :param t_lim: The maximum time step to run this simulation.
        :param store_freq: Frequency to store a snapshot of the grid, which can be
        used later to make cool stuff.
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
            u = self.manager.chooseSimulationStep(SimulationContext(model))
            self.height[u] += 1
            self.t += 1

            # The step is concluded, so make the step callbacks.
            step_context = SimulationStepContext(self, u, False)
            self.manager.onSimulationStep(step_context)
            for listener in self.listeners:
                listener.onSimulationStep(step_context)

            # If toppling is triggered, performs them in a sub-routine.
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

            u = queue.popleft()
            self.t += 1

            # Perform reset and spread the sand on cell (i, j).
            self.height[u] -= self.topple_limit

            # If this is not enough, adds the cell back into the queue.
            if self.height[u] >= self.topple_limit:
                print("Wha? The topology must be weird. Consider lifting the "
                      "topple limit.")
                queue.append(u)

            # Gets the neighbour coordinate.
            for v in self.graph.neighbors(u):
                # If the new cell is within the boundary, give it sand; if that makes it full,
                # adds it into the topple search stack; otherwise, let it drop into the void
                if v != self.sink_index:
                    self.height[v] += 1

                    # If the new cell is full, adds it into the queue and perform toppling on it.
                    # Avoids repeated queuing by enforcing equality.
                    if self.height[v] == self.topple_limit:
                        queue.append(v)

            # The step is concluded, so make the callbacks.
            step_context = SimulationStepContext(self, u, True)
            self.manager.onSimulationStep(step_context)
            for listener in self.listeners:
                listener.onSimulationStep(step_context)

    def visualise(self, *args, **kwargs):
        """ A shortcut to the visualise function. """
        return visualise(self.original_graph, self.height, *args, **kwargs)

    def visualise_as_grid(self, n: int, m: int, *args, **kwargs):
        """ A shortcut to the visualise_as_grid function. """
        return visualise_as_grid(n, m, self.height)


def make_grid_graph(m: int, n: int):
    """ Makes a graph with a grid-like topology.

    :param m: Number of rows in the grid.
    :param n: Number of columns in the grid.
    :return: The grid graph along with the set of boundary vertices.
    """
    edges = ([(j * m + i, j * m + i + 1) for i in range(n - 1) for j in range(m)]
             + [(j * m + i, (j + 1) * m + i) for i in range(n) for j in range(m - 1)])
    graph = ig.Graph(edges)
    boundary_vertices = set(j * m + i for i in (0, n - 1) for j in (0, m - 1))
    return graph, boundary_vertices


def visualise(
        graph: ig.Graph, height: list[int], max_height=None,
        fig: matplotlib.figure.Figure = None,
        ax: matplotlib.figure.Axes = None):
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


def visualise_as_grid(
        m: int, n: int, height: list[int], max_height: int = None,
        fig: matplotlib.figure.Figure = None, ax: matplotlib.figure.Axes = None,
        xticks=None, yticks=None, labels=None, title=None):
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
    :param xticks: x-axis labels.
    :param yticks: y-axis labels.
    :param labels: Grid cell annotations.
    :param title: Plot title.
    :return: Figure, axes pair with the graph plotted.
    """
    # Gridify the graph.
    grid = [[height[i * m + j] for j in range(n)] for i in range(m)]

    # Create the awesome plot.
    if fig is None or ax is None:
        fig, ax = plt.subplots()

    ax.imshow(grid, cmap="Greys", vmin=0, vmax=4, interpolation="none")

    # Show ticks and label them with the respective entries.
    if xticks is not None:
        if xticks == "auto":
            xticks = [str(x) for x in range(model.n)]
        ax.set_xticks(np.arange(len(xticks)), labels=xticks)
    if yticks is not None:
        if yticks == "auto":
            yticks = [str(x) for x in range(model.m)]
        ax.set_yticks(np.arange(len(yticks)), labels=yticks)

    # # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    if labels is not None:
        if labels == "auto":
            labels = [[str(x) for x in l] for l in grid]

        for i in range(m):
            for j in range(n):
                ax.text(j, i, labels[i][j], ha="center", va="center")

    # Set the title.
    ax.set_title(title)
    fig.tight_layout()

    # Return the figure and user can do whatever they want with it.
    return fig, ax


def save_video(graph: ig.Graph, history: list[list[int]], max_height=None):
    """ Save the transition as a video. """
    if max_height is None:
        max_height = max(max(height) for height in history)

    # Setup the video
    frames_count = len(history)
    duration = 30
    fig, ax = plt.subplots()

    # Helper function to make frames.
    def make_frame(t):
        # Clear the axes before adding the next plot.
        ax.clear()
        visualise(graph, history[round(t * (frames_count - 1) / duration)])
        return mplfig_to_npimage(fig)

    # creating animation
    animation = VideoClip(make_frame=make_frame, duration=duration)
    animation.write_videofile("./out/video.mp4", fps=20, loop=True)
    # autoplay=True


if __name__ == "__main__":
    # Testing.
    graph, boundary_vertices = make_grid_graph(10, 10)

    model = Model(
        graph=graph,
        boundary_vertices=list(boundary_vertices)
    )
    model.simulate()
    fig, ax = model.visualise()
    fig.savefig("./out/1.jpg")
