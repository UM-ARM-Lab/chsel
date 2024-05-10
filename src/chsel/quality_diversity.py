import abc
from typing import Optional
import logging

import cma
import numpy as np

import torch
from arm_pytorch_utilities.tensor_utils import ensure_tensor

from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter, GradientArborescenceEmitter
from ribs.schedulers import Scheduler
from chsel.costs import RegistrationCost
from chsel.types import SimilarityTransform, ICPSolution
from chsel.measure import PositionMeasure
from chsel import registration_util

logger = logging.getLogger(__name__)


# try measure function on the rotation dimensions


class QDOptimization:
    def __init__(self, registration_cost: RegistrationCost,
                 model_points_world_frame: torch.tensor,
                 measure=None,
                 init_transform: Optional[SimilarityTransform] = None,
                 sigma=0.1,
                 num_emitters=5,
                 save_loss_plot=False,
                 savedir=registration_util.ROOT_DIR,
                 **kwargs):
        """

        :param registration_cost: some implementation of Equation 6
        :param model_points_world_frame: not actually used, but useful for tracking and debugging registration process
        can be points with known SDF value = 0 (surface points) in the world frame
        :param init_transform: transform from which to start the estimation
        :param sigma: QD parameter for specifying degree of exploration
        :param num_emitters: number of points to consider simultaneously, for example if the search space is large
        :param save_loss_plot: whether to plot losses and save them
        :param savedir: where to save the plotted losses
        :param kwargs: kwargs forwarded to creating the scheduler
        """
        self.registration_cost = registration_cost
        self.X = model_points_world_frame
        self.Xt = self.X
        self.B = self.Xt.shape[0]

        self.init_transform = init_transform
        self.sigma = sigma
        self.save_loss_plot = save_loss_plot
        self.savedir = savedir

        self.device = self.Xt.device
        self.dtype = self.Xt.dtype

        self.measure = measure
        if self.measure is None:
            self.measure = PositionMeasure(2, dtype=self.dtype, device=self.device)

        Xt, R, T, s = registration_util.apply_init_transform(self.Xt, self.init_transform)
        x = self.measure.get_numpy_x(R, T)
        self.num_emitters = num_emitters
        self.scheduler = self.create_scheduler(x, **kwargs)

    def run(self):
        Xt, R, T, s = registration_util.apply_init_transform(self.Xt, self.init_transform)

        # initialize the transformation history
        t_history = []
        losses = []

        while not self.is_done():
            cost = self.step()

            losses.append(cost)

        R, T, rmse = self.process_final_results(None, losses)

        return ICPSolution(True, rmse, Xt, SimilarityTransform(R, T, s), t_history)

    @abc.abstractmethod
    def add_solutions(self, solutions):
        pass

    @abc.abstractmethod
    def create_scheduler(self, x0, *args, **kwargs):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def process_final_results(self, s, losses):
        pass

    @abc.abstractmethod
    def is_done(self):
        return False

    @abc.abstractmethod
    def get_all_elite_solutions(self):
        return None


class CMAES(QDOptimization):
    def create_scheduler(self, x, *args, **kwargs):
        x0 = x[0]
        options = {"popsize": self.B, "seed": np.random.randint(0, 10000), "tolfun": 1e-5, "tolfunhist": 1e-6}
        options.update(kwargs)
        es = cma.CMAEvolutionStrategy(x0=x0, sigma0=self.sigma, inopts=options)
        return es

    def is_done(self):
        return self.scheduler.stop()

    def step(self):
        solutions = self.scheduler.ask()
        # convert back to R, T, s
        R, T = self.measure.get_torch_RT(np.stack(solutions))
        cost = self.registration_cost(R, T, None)
        self.scheduler.tell(solutions, cost.cpu().numpy())
        return cost

    def add_solutions(self, solutions):
        pass

    def get_all_elite_solutions(self):
        return None

    def process_final_results(self, s, losses):
        # convert ES back to R, T
        solutions = self.scheduler.ask()
        R, T = self.measure.get_torch_RT(np.stack(solutions))
        rmse = self.registration_cost(R, T, s)

        if self.save_loss_plot:
            registration_util.plot_poke_losses(losses, savedir=self.savedir)

        return R, T, rmse


class CMAME(QDOptimization):
    def __init__(self, *args, bins=20, iterations=100,
                 # require an explicit range
                 ranges=None,
                 outlier_ratio=5.0,  # reject solutions that are this many times worse than the best
                 outlier_absolute_tolerance=1e-6,  # handle rmse = 0 by allowing for at least this much tolerance
                 qd_score_offset=-100,  # useful for tracking the archive QD score as monotonically increasing
                 measure=None,
                 **kwargs):
        if "sigma" not in kwargs:
            kwargs["sigma"] = 1.0
        self.measure = measure
        if self.measure is None:
            self.measure = PositionMeasure(2)

        if isinstance(bins, (float, int)):
            self.bins = [bins for _ in range(self.measure.measure_dim)]
        else:
            assert len(bins) == self.measure.measure_dim
            self.bins = bins
        self.iterations = iterations
        self.ranges = ranges
        self.qd_score_offset = qd_score_offset
        self.outlier_ratio = outlier_ratio
        self.outlier_absolute_tolerance = outlier_absolute_tolerance

        self.archive = None
        self.i = 0
        self.qd_scores = []

        super(CMAME, self).__init__(*args, measure=self.measure, **kwargs)

    def _create_ranges(self):
        if self.ranges is None:
            raise RuntimeError("An explicit archive range must be specified")

    def create_scheduler(self, x, *args, **kwargs):
        self._create_ranges()
        self.archive = GridArchive(solution_dim=x.shape[1], dims=self.bins,
                                   ranges=self.ranges[:self.measure.measure_dim],
                                   seed=np.random.randint(0, 10000), qd_score_offset=self.qd_score_offset)
        emitters = [
            EvolutionStrategyEmitter(self.archive, x0=x[i], sigma0=self.sigma, batch_size=self.B,
                                     seed=np.random.randint(0, 10000)) for i in
            range(self.num_emitters)
        ]
        scheduler = Scheduler(self.archive, emitters)
        return scheduler

    def is_done(self):
        return self.i >= self.iterations

    def step(self):
        self.i += 1
        solutions = self.scheduler.ask()
        # evaluate the models and record the objective and behavior
        # note that objective is -cost
        R, T = self.measure.get_torch_RT(np.stack(solutions))
        cost = self.registration_cost(R, T, None)
        bcs = self.measure(solutions)
        self.scheduler.tell(-cost.cpu().numpy(), bcs)
        qd = self.archive.stats.norm_qd_score
        self.qd_scores.append(qd)
        logger.debug("step %d norm QD score: %f", self.i, qd)
        return cost

    def _add_solutions(self, solutions):
        R, T = self.measure.get_torch_RT(np.stack(solutions))
        rmse = self.registration_cost(R, T, None)
        self.archive.add(solutions, -rmse.cpu().numpy(), self.measure(solutions))

    def add_solutions(self, solutions):
        if solutions is None:
            return
        assert isinstance(solutions, np.ndarray)
        SOLUTION_CHUNK = 300
        for i in range(0, solutions.shape[0], SOLUTION_CHUNK):
            self._add_solutions(solutions[i:i + SOLUTION_CHUNK])

    def get_all_elite_solutions(self):
        return self.archive.data('solution')

    def process_final_results(self, s, losses):
        objectives = self.archive.data('objective')
        all_solutions = self.archive.data('solution')

        cost = -objectives
        # filter out all solutions that are more than outlier_ratio times worse than the best
        lowest_cost = np.min(cost)
        inlier_mask = cost < lowest_cost * self.outlier_ratio + self.outlier_absolute_tolerance
        solutions = all_solutions[inlier_mask]
        cost = cost[inlier_mask]
        # rather than min, resort to elites
        if len(solutions) == 0:
            cost = -objectives
            elite_cost = np.quantile(cost, 0.01)
            inlier_mask = cost < elite_cost * self.outlier_ratio
            solutions = all_solutions[inlier_mask]
            cost = cost[inlier_mask]
        # if no elites, then resort to actual min
        if len(solutions) == 0:
            cost = -objectives
            lowest_cost = np.min(cost)
            solutions = all_solutions[cost == lowest_cost]
            solutions = solutions.reshape(1, -1)
        if len(solutions) > self.B:
            order = np.argpartition(cost, self.B)
            solutions = solutions[order[:self.B]]
        # if there are fewer than B solutions, randomly sample the remaining ones from existing solutions
        elif len(solutions) < self.B:
            # sample from existing solutions with replacement by index
            resampled_indices = np.random.choice(np.arange(len(solutions)), self.B - len(solutions))
            solutions = np.concatenate([solutions, solutions[resampled_indices]], axis=0)
        # convert back to R, T
        R, T = self.measure.get_torch_RT(solutions)
        rmse = self.registration_cost(R, T, s)

        if self.save_loss_plot:
            registration_util.plot_poke_losses(losses, savedir=self.savedir)
            qd_scores = [torch.tensor(v).view(1) for v in self.qd_scores]
            registration_util.plot_poke_losses(qd_scores, savedir=self.savedir, loss_name='qd_score', logy=False,
                                               ylabel='norm qd score')
            registration_util.plot_qd_archive(self.archive, savedir=self.savedir)

        return R, T, rmse


class CMAMEGA(CMAME):
    def __init__(self, *args, lr=0.05, **kwargs):
        self.lr = lr
        super(CMAMEGA, self).__init__(*args, **kwargs)

    def create_scheduler(self, x, *args, **kwargs):
        self._create_ranges()
        self.archive = GridArchive(solution_dim=x.shape[1], dims=self.bins, seed=np.random.randint(0, 10000),
                                   ranges=self.ranges[:self.measure.measure_dim], qd_score_offset=self.qd_score_offset)
        emitters = []
        # emitters += [
        #     EvolutionStrategyEmitter(self.archive, x0=x[i], sigma0=self.sigma, batch_size=self.B) for i in
        #     range(self.num_emitters)
        # ]
        # rb = 3
        # rot_bounds = np.array([[-rb, rb] for _ in range(6)])
        # bounds = np.concatenate((self.ranges, rot_bounds))
        emitters += [
            GradientArborescenceEmitter(self.archive, x0=x[i], sigma0=self.sigma, lr=self.lr, grad_opt="adam",
                                        selection_rule="filter", bounds=None, batch_size=self.B - 1,
                                        seed=np.random.randint(0, 10000)) for i in
            range(self.num_emitters)
        ]
        scheduler = Scheduler(self.archive, emitters)
        return scheduler

    def _f(self, x):
        R, T = self.measure.get_torch_RT(x)
        return self.registration_cost(R, T, None)

    def step(self):
        solutions = self.scheduler.ask_dqd()
        bcs = self.measure(solutions)
        # evaluate the models and record the objective and behavior
        # note that objective is -cost
        # get objective gradient and also the behavior gradient
        x = ensure_tensor(self.device, self.dtype, solutions)
        x.requires_grad = True
        cost = self._f(x)
        cost.sum().backward()
        objective_grad = -(x.grad.cpu().numpy())
        objective = -cost.detach().cpu().numpy()
        objective_grad = objective_grad.reshape(x.shape[0], 1, -1)
        measure_grad = self.measure.grad(x)

        jacobian = np.concatenate((objective_grad, measure_grad), axis=1)
        self.scheduler.tell_dqd(objective, bcs, jacobian)

        return super(CMAMEGA, self).step()


def initialize_qd_archive(T, rmse, measure_fn, outlier_ratio=5.0, outlier_absolute_tolerance=0.1, range_sigma=3,
                          min_std=1e-4):
    x = measure_fn.get_numpy_x(T[:, :3, :3], T[:, :3, 3])
    measure = measure_fn(x)

    # ensure the measure is 2D
    measure = measure.reshape(-1, measure_fn.measure_dim)

    # filter out any solution that is above outlier_ratio of the best solution found
    keep = rmse < (rmse.min() * outlier_ratio + outlier_absolute_tolerance)
    m = measure[keep.cpu()]
    logger.info(f"keep {len(m)} solutions out of {len(measure)} for QD initialization with min rmse {rmse.min()}")

    centroid, m_std = measure_fn.compute_moments(m)
    m_std = np.maximum(m_std, min_std)

    ranges = np.array((centroid - m_std * range_sigma, centroid + m_std * range_sigma)).T
    return ranges.reshape(measure_fn.measure_dim, 2)
