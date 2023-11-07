class BaseGrid(MutableMapping[K, T]):
    @abstractmethod
    def neighbours(self, u: K) -> list[K]:
        """ Retrieves the list of indices neighbouring to u.

        :param u: Key to the item.
        """
        pass


class ArrayGrid(BaseGrid[K, T]):
    ADJ: list[tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    n: int
    m: int
    grid: list[list[int]]
    vertex_data: dict[K, T]

    def __init__(self, n: int, m: int = None):
        if m is None:
            self.m = n
        self.n = n

    def neighbours(self, u: K) -> list[K]:
        """ Retrieves the list of indices neighbouring to u. """
        i = u[0]
        j = u[1]
        result = []
        if i < 0 or i >= self.n or j < 0 or j >= self.m:
            raise ValueError(f"Index out of range: ({i}, {j})")

        for di, dj in self.ADJ:
            ni = u[0] + di
            nj = u[1] + dj
            if 0 <= ni < self.n and 0 <= nj < self.m:
                result.append((ni, nj))
        return result

    def __getitem__(self, key: K):
        return self.grid[key[0]][key[1]]

    def __setitem__(self, key: K, value: T):
        self.grid[key[0]][key[1]] = value

    def __delitem__(self, key: K) -> None:
        self.grid[key[0]][key[1]] = 0

    def __len__(self) -> int:
        return self.m * self.n

    def __iter__(self):
        return chain(self.grid)


class GraphGrid(BaseGrid[K, T]):
    graph: ig.Graph
    sink_index: int
    boundary_vertices: list[int]

    def __init__(self, graph: ig.Graph, boundary_vertices: list[int]):
        self.graph = graph
        self.sink_index = self.graph.vcount()
        self.boundary_vertices = boundary_vertices
        self.graph.add_vertices(1)
        self.graph.add_edges([(u, self.sink_index) for u in self.boundary_vertices])

    def neighbours(self, u: K) -> list[K]:
        """ Retrieves the list of indices neighbouring to u. """
        return self.graph.neighbors(u)

    def is_sink(self, u: K) -> bool:
        """ Returns whether the item u is a sink. """
        return u == self.sink_index


@dataclass
class LegacyModel(object):
    """ The model. """
    n: int = 100
    m: int = 100
    topple_limit: int = 4
    t_lim: int = 1_000_000
    t: int = 0
    store_freq: int = 1_000_000
    hard_stop: bool = False
    grid: Grid[int] = field(init=False)
    history: list[Grid[int]] = field(init=False)

    def __post_init__(self):
        self.history = []
        self.grid = [[0 for _ in range(self.n)] for _ in range(self.m)]

    def _topple(self, original_i: int, original_j: int):
        """ Non-recursive!

        :param original_i: The starting point of toppling.
        :param original_j: The starting point of toppling.
        :param a: The grid reference.
        :param track_topple: Whether to track toppling.
        :return: The time it takes to complete the toppling.
        """
        queue = deque([(original_i, original_j)])

        while len(queue) > 0:
            if self.t == self.t_lim and self.hard_stop:
                print("Oh no, the toppling is abruptly stopped due to 'hard_stop' being set to True."
                      "If you don't mind waiting a couple more milliseconds, please set it to False.")
                return

            i, j = queue.popleft()
            self.t += 1

            # Make a snapshot if required.
            if self.t % self.store_freq == 0:
                self.history.append([x.copy() for x in self.grid])

            # Perform reset and spread the sand on cell (i, j).
            self.grid[i][j] -= self.topple_limit

            # If this is not enough, adds the cell back into the queue.
            if self.grid[i][j] >= self.topple_limit:
                print("Wha? The topology must be weird. Consider lifting the topple limit.")
                queue.append((i, j))

            for di, dj in ADJ:
                # Gets the neighbour coordinate.
                ni = i + di
                nj = j + dj

                # If the new cell is within the boundary, give it sand; if that makes it full,
                # adds it into the topple search stack; otherwise, let it drop into the void
                if 0 <= ni < self.n and 0 <= nj < self.m:
                    self.grid[ni][nj] += 1

                    # If the new cell is full, adds it into the queue and perform toppling on it.
                    # Avoids repeated queuing by enforcing equality.
                    if self.grid[ni][nj] == self.topple_limit:
                        queue.append((ni, nj))

    def simulate(self, t_lim: int = 10_000_000, store_freq: int = 10_000_000,
                 hard_stop: bool = False):
        """ Performs the simulation.

        :param model: The simulation model.
        :param t_lim: The maximum time step to run this simulation.
        :param store_freq: Frequency to store a snapshot of the grid, which can be
        used later to make cool stuff.
        :param hard_stop: Whether the time limit is strict. If True then the simulation
        will stop exactly at t_lim and unstable sand cells may exist; otherwise,
        the simulation stops after the current toppling ends.
        """
        # Pre-compute using the power of vectorisation.
        rand_i = np.random.randint(0, self.n, t_lim)
        rand_j = np.random.randint(0, self.m, t_lim)

        # Stores the initial grid, regardless. A deep copy is necessary.
        self.history.append([x.copy() for x in self.grid])

        while self.t < t_lim:
            # TODO: One way to improve it, add a callback for generating i and j.
            # i = rand_i[t]
            # j = rand_j[t]
            i = 49
            j = 49
            self.grid[i][j] += 1
            self.t += 1

            # Stores the grid if required.
            if self.t % self.store_freq == 0:
                self.history.append([x.copy() for x in self.grid])

            # Performs the toppling iterations in a sub-routine.
            if self.grid[i][j] >= self.topple_limit:
                self._topple(i, j)

        return self

    def save_all(self):
        """ Save everything1 """
        for i, a in enumerate(self.history):
            self.visualise_and_save(a, f"./out/{i}.jpg")

    def visualise_grid(self, grid: Grid[int],
                       xticks: Optional[list[str] | Literal["auto"]] = None,
                       yticks: Optional[list[str] | Literal["auto"]] = None,
                       labels: Optional[Grid[str] | Literal["auto"]] = None,
                       title: str = "Grid"):
        """ Produce an image of the grid.


        """
        # Just some type annotations so the IDE is not confused.
        fig: matplotlib.figure.Figure
        ax: matplotlib.figure.Axes

        # Create the awesome plot.
        fig, ax = plt.subplots()
        ax.imshow(self.grid, cmap="Greys", vmin=0, vmax=4, interpolation="none")

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

            for i in range(model.m):
                for j in range(model.n):
                    ax.text(j, i, labels[i][j], ha="center", va="center")

        # Set the title.
        ax.set_title(title)
        fig.tight_layout()

        # Return the figure and user can do whatever they want with it.
        return fig

    def save_video(self):
        # numpy array
        x = np.linspace(-2, 2, 200)

        # duration of the video
        duration = 2

        # matplot subplot
        fig, ax = plt.subplots()

        # method to get frames
        def make_frame(t):
            # clear
            ax.clear()

            # plotting line
            ax.plot(x, np.sinc(x ** 2) + np.sin(x + 2 * np.pi / duration * t), lw=3)
            ax.set_ylim(-1.5, 2.5)

            # returning numpy image
            return mplfig_to_npimage(fig)

        # creating animation
        animation = VideoClip(make_frame=make_frame, duration=duration)

        # displaying animation with auto play and looping
        animation.write_videofile("./out/video.mp4", fps=20, loop=True, autoplay=True)
