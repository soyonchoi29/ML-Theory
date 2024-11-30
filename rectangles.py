import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import seaborn as sns


################################################################################################################
###################                             PART 1: THE ALGORITHM                        ###################
################################################################################################################


class AAR:
    def __init__(self, d):
        self.d = d
        self.lower_bounds = np.ones(shape=(1, d))  # holds lower bound of rectangle in each dimension d
        self.upper_bounds = -1 * np.ones(shape=(1, d))  # holds upper bound of rectangle in each dimension d

    def fit(self, X, Y):
        # X = (N, d), U([-1, 1]^d)
        # Y = (N, 1), {-1, 1}

        # if no positive points in Y, return the "always -" classifier
        if not np.isin(1, Y):
            # will return -1 since nothing is above 1 and below -1
            self.lower_bounds = np.ones(shape=(1, self.d))
            self.upper_bounds = -1 * np.ones(shape=(1, self.d))
            return self

        # only consider the points that are labeled as positive (inside the box)
        X = X[Y[:, 0]==1]

        # get the min and max X values in each dimension to draw the smallest box possible
        min_X = np.min(X, axis=0)
        min_X = np.expand_dims(min_X, axis=0)  # min_X = (1, d)
        max_X = np.max(X, axis=0)
        max_X = np.expand_dims(max_X, axis=0)  # max_X = (1, d)

        assert min_X.shape == self.lower_bounds.shape
        assert max_X.shape == self.upper_bounds.shape

        self.lower_bounds = np.minimum(self.lower_bounds, min_X)
        self.upper_bounds = np.maximum(self.upper_bounds, max_X)

        return self

    def forward(self, X):
        # input: X = (N, d)
        assert X.shape[1] == self.d  # check input dimensions
        N = X.shape[0]

        # dimension-wise bounds check
        lower = np.repeat(self.lower_bounds, N, axis=0)
        upper = np.repeat(self.upper_bounds, N, axis=0)
        res = np.logical_and((X >= lower), (X <= upper))

        # if x fits in bounds in all dimensions (it's in the rectangle), it's positive!
        # otherwise, it's negative :(
        res = np.all(res, axis=1)
        res = np.reshape(res, newshape=(N, 1)).astype(int)
        res[res[:]==0]=-1

        return res


################################################################################################################
###################                             DATA GENERATOR CLASS                         ###################
################################################################################################################


class Data:
    def __init__(self, d, sampler, positive='half'):
        self.d = d
        self.sampler = sampler(d)
        self.positive = positive
        self.upper_bound = 1.0/(2**(1/d))
        self.lower_bound = -1 * self.upper_bound

    def f(self, X):
        N = X.shape[0]
        res = np.ones((N, 1))

        if self.positive == 'all':
            return res
        elif self.positive == 'none':
            return -1 * res
        else:  # if self.positive == 'half' (default)
            # make dims (N, d) for comparison with input X
            lower = np.repeat(self.lower_bound, N, axis=0)
            lower = np.expand_dims(lower, axis=1)
            lower = np.repeat(lower, self.d, axis=1)

            upper = np.repeat(self.upper_bound, N, axis=0)
            upper = np.expand_dims(upper, axis=1)
            upper = np.repeat(upper, self.d, axis=1)

            # + if -sqrt(2)/2 <= x1, x2 <= sqrt(2)/2
            # - otherwise
            res = np.logical_and((X >= lower), (X <= upper))
            res = np.all(res, axis=1)
            res = np.reshape(res, newshape=(N, 1)).astype(int)
            res[res[:]==0]=-1

            return res

    def sample(self, N):
        X = self.sampler.sample(N)
        Y = self.f(X)
        return X, Y


################################################################################################################
###################                         SAMPLERS FOR X = [-1, 1]^d                       ###################
################################################################################################################


class Sampler:
    def __init__(self, d):
        self.d = d

    def sample(self, N):
        pass


class UniformSampler(Sampler):
    def sample(self, N):
        return np.random.uniform(-1.0, 1.0, (N, self.d))


class GaussianSampler(Sampler):
    def sample(self, N):
        return np.random.normal(0.0, 0.5, (N, self.d))


################################################################################################################
###################                          FUNCTIONS FOR COMPUTING ERROR                   ###################
################################################################################################################


def get_error(model, sampler, positive):
    if sampler == UniformSampler and positive == 'half':
        return uniform_error(model)
    else:
        return test_error(100, model, Data(model.d, sampler, positive))


def uniform_error(model):
    area = np.prod(np.subtract(model.upper_bounds, model.lower_bounds))
    error = ((2**(d-1)) - area) / (2**d)
    return error


def test_error(num_test, model, data):
    X_test, Y_test = data.sample(num_test)
    Y_pred = model.forward(X_test)
    error = np.sum(1*(Y_pred != Y_test)) / Y_test.size
    return error


################################################################################################################
###################        Helper functions for computing an empirical sample complexity     ###################
################################################################################################################


def get_empirical_bounds(d, T, epsilons, deltas, start_n_s, sampler, positive='half'):
    # returns empirical number of samples n required to learn a probably approximately correct model
    emp_n_s = np.zeros(shape=start_n_s.shape)
    for i in range(start_n_s.shape[0]):
        for j in range(start_n_s.shape[1]):
            emp_n_s[i, j] = binary_search(0, int(start_n_s[i, j]), d, T, epsilons[j], deltas[i], sampler, positive)
    return emp_n_s


def binary_search(lo, hi, d, T, epsilon, delta, sampler, positive):
    if hi <= lo:
        return lo
    mid = (hi+lo)//2
    if check_PAC(mid, d, T, epsilon, delta, sampler, positive):
        return binary_search(lo, mid, d, T, epsilon, delta, sampler, positive)
    else:
        return binary_search(mid+1, hi, d, T, epsilon, delta, sampler, positive)


def check_PAC(n, d, T, epsilon, delta, sampler, positive):
    # given sample of n points, check if algo. learns probably approximately correct model on H
    data = Data(d, sampler, positive)
    times_corr_enough = 0
    for t in range(T):
        X, Y = data.sample(n)
        model = AAR(d)
        model = model.fit(X, Y)

        error = get_error(model, sampler, positive)

        if error <= epsilon:
            times_corr_enough += 1
    if (times_corr_enough / T) >= 1 - delta:
        return True
    else:
        return False


################################################################################################################
###################    Functions that compute and return the different theoretical bounds    ###################
################################################################################################################


def class_bound(epsilons, deltas, d):
    num_epsilons = epsilons.size
    num_deltas = deltas.size
    epsilons = np.repeat(np.expand_dims(epsilons, axis=0), num_deltas, axis=0)  # (num_deltas, num_epsilons)
    deltas = np.repeat(deltas, num_epsilons, axis=0).reshape((num_deltas, num_epsilons))

    epsilons = np.repeat(np.expand_dims(epsilons, axis=0), d.size, axis=0)  # (num_epsilons*num_d, num_epsilons)
    deltas = np.repeat(np.expand_dims(deltas, axis=0), d.size, axis=0)
    d_s = np.repeat(np.repeat(d, num_epsilons), num_deltas).reshape(epsilons.shape)

    return np.log(deltas / (2.0 * d_s)) / np.log(1 - (epsilons / (2.0 * d_s)))


def book_bound(epsilons, deltas, d):
    num_epsilons = epsilons.size
    num_deltas = deltas.size
    epsilons = np.repeat(np.expand_dims(epsilons, axis=0), num_deltas, axis=0)  # (num_deltas, num_epsilons)
    deltas = np.repeat(deltas, num_epsilons, axis=0).reshape((num_deltas, num_epsilons))

    epsilons = np.repeat(np.expand_dims(epsilons, axis=0), d.size, axis=0)  # (num_epsilons*num_d, num_epsilons)
    deltas = np.repeat(np.expand_dims(deltas, axis=0), d.size, axis=0)
    d_s = np.repeat(np.repeat(d, num_epsilons), num_deltas).reshape(epsilons.shape)

    return ((2.0 * d_s) / epsilons) * np.log((2.0 * d_s) / deltas)


################################################################################################################
################################################################################################################


if __name__ == '__main__':


    ##########################################################################################################
    # PART 2. MAKE SOME HEATMAPS
    ##########################################################################################################


    step = 0.05
    epsilons = np.arange(step, 1.0, step=step)
    deltas = np.arange(step, 1.0, step=step)

    # combs = np.array(np.meshgrid(epsilons, deltas)).T.reshape(-1, 2)
    # combs = np.reshape(combs, (len(epsilons), len(deltas), combs.shape[1]))  # (n_epsilons, n_deltas)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.set_tight_layout({"pad": 1})

    # 1. the bound we derived in class: n >= log(delta/4)/log(1-epsilon/4)
    class_bounds = np.squeeze(class_bound(epsilons, deltas, np.array([[2]])))
    sns.heatmap(class_bounds,
                cmap=sns.cubehelix_palette(as_cmap=True),
                ax=axs[0],
                xticklabels=[str(epsilon)[:3] for epsilon in epsilons],
                yticklabels=[str(delta)[:3] for delta in deltas])
    axs[0].set_title('Bound derived in class:')
    axs[0].set_xlabel('Epsilon')
    axs[0].set_xticklabels([str(epsilon)[:4] for epsilon in epsilons], rotation=45, ha='right', fontdict={'size': 8})
    axs[0].set_ylabel('Delta')
    axs[0].set_yticklabels([str(delta)[:4] for delta in deltas], rotation=0, ha='right', fontdict={'size': 8})

    # 2. another bound commonly used (see problem 2.2 in the book)
    book_bounds = np.squeeze(book_bound(epsilons, deltas, np.array([[2]])))
    sns.heatmap(book_bounds,
                cmap=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True),
                ax=axs[1],
                xticklabels=[str(epsilon)[:3] for epsilon in epsilons],
                yticklabels=[str(delta)[:3] for delta in deltas])
    axs[1].set_title('Bound commonly used:')
    axs[1].set_xlabel('Epsilon')
    axs[1].set_xticklabels([str(epsilon)[:4] for epsilon in epsilons], rotation=45, ha='right', fontdict={'size': 8})
    axs[1].set_ylabel('Delta')
    axs[1].set_yticklabels([str(delta)[:4] for delta in deltas], rotation=0, ha='right', fontdict={'size': 8})

    # 3. empirically test the lowest tr set size n s.t. err < epsilon w prob >= 1-delta
    d = 2  # for now, we in R2
    T = 100  # let's run 100 trials per n
    n_s = np.minimum(class_bounds, book_bounds)  # = (num_tested_epsilons, num_tested_deltas)
    emp_n_s = get_empirical_bounds(d, T, epsilons, deltas, n_s, sampler=UniformSampler)

    sns.heatmap(emp_n_s,
                cmap=sns.color_palette("light:b", as_cmap=True),
                ax=axs[2],
                xticklabels=[str(epsilon)[:3] for epsilon in epsilons],
                yticklabels=[str(delta)[:3] for delta in deltas])
    axs[2].set_title('Empirically tested bound:')
    axs[2].set_xlabel('Epsilon')
    axs[2].set_xticklabels([str(epsilon)[:4] for epsilon in epsilons], rotation=45, ha='right', fontdict={'size': 8})
    axs[2].set_ylabel('Delta')
    axs[2].set_yticklabels([str(delta)[:4] for delta in deltas], rotation=0, ha='right', fontdict={'size': 8})

    plt.show()


    ##########################################################################################################
    # PART 3. MUCK WITH d
    ##########################################################################################################


    # how to quantify how far the empirical bound is from the theoretical?
    # get difference for some reasonable pair (epsilon, delta)
    # epsilon, delta = 0.1, 0.1

    # scratch that, since we have extra time i decided to add more functionality to this plot :)
    max_d = 30
    d_s = np.arange(1, max_d+1)  # array from 1 to max_d
    T = 50  # let's run 50 trials per n

    all_emp_n_s_uniform = []


    ##########################################################################################################
    # PART 4: MUCK WITH D
    # Note: Plot for Part 3 has been combined with the plot for Part 4.
    ##########################################################################################################


    all_emp_n_s_normal = []

    class_bounds = class_bound(epsilons, deltas, d_s)
    book_bounds = book_bound(epsilons, deltas, d_s)

    for d in d_s:
        print("Currently at dimension d =", d)
        emp_n_s_uniform = get_empirical_bounds(d, T, epsilons, deltas, class_bounds[d-1], sampler=UniformSampler)
        emp_n_s_normal = get_empirical_bounds(d, T, epsilons, deltas, class_bounds[d-1], sampler=GaussianSampler)
        all_emp_n_s_uniform.append(emp_n_s_uniform)
        all_emp_n_s_normal.append(emp_n_s_normal)

    all_emp_n_s_uniform = np.stack(all_emp_n_s_uniform, axis=0)  # (max_d, num_deltas, num_epsilons)
    all_emp_n_s_normal = np.stack(all_emp_n_s_normal, axis=0)  # (max_d, num_deltas, num_epsilons)

    diff_class_uniform = np.abs(class_bounds-all_emp_n_s_uniform)/class_bounds  # if v diff, trends toward 1
    diff_book_uniform = np.abs(book_bounds-all_emp_n_s_uniform)/book_bounds
    diff_class_normal = np.abs(class_bounds-all_emp_n_s_normal)/class_bounds
    diff_book_normal = np.abs(book_bounds-all_emp_n_s_normal)/book_bounds

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots()
    ax.set_ylim((0, 1.0))
    i, j = 0, 0  # start w epsilon, delta = 0.1

    # plot d vs. diff in bounds
    # diff btwn empirical (uniform dist) and bound from class
    line0, = ax.plot(d_s, diff_class_uniform[:,j,i].flatten(), label='d vs. |class-empirical|/class (U)')
    # diff btwn empirical (uniform dist) and bound from book
    line1, = ax.plot(d_s, diff_book_uniform[:,j,i].flatten(), label='d vs. |book-empirical|/book (U)')
    # diff btwn empirical (normal dist) and bound from class
    line2, = ax.plot(d_s, diff_class_normal[:,j,i].flatten(), label='d vs. |class-empirical|/class (N)')
    # diff btwn empirical (normal dist) and bound from book
    line3, = ax.plot(d_s, diff_book_normal[:,j,i].flatten(), label='d vs. |book-empirical|/book (N)')

    ax.set_title('d vs. observed diff in bounds (uniform, normal)')
    ax.set_xlabel('d')
    ax.set_ylabel('bound diff')
    ax.legend(loc='best')

    # make room for sliders!
    fig.subplots_adjust(left=0.20, bottom=0.30)

    # slider for epsilon
    ax_epsilon = fig.add_axes([0.25, 0.15, 0.65, 0.03])
    epsilon_slider = Slider(
        ax=ax_epsilon,
        label='Epsilon',
        valinit=epsilons[0],
        valmin=epsilons[0],
        valmax=deltas[-1],
        valstep=epsilons,
    )

    # slider for delta
    ax_delta = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    delta_slider = Slider(
        ax=ax_delta,
        label='Delta',
        valinit=deltas[0],
        valmin=deltas[0],
        valmax=deltas[-1],
        valstep=deltas
    )

    # function to update plotted line based on slider value
    def update(val):
        curr_epsilon = epsilon_slider.val
        i = np.where(epsilons==curr_epsilon)[0][0]
        curr_delta = delta_slider.val
        j = np.where(deltas==curr_delta)[0][0]

        line0.set_ydata(diff_class_uniform[:,j,i].flatten())
        line1.set_ydata(diff_book_uniform[:,j,i].flatten())
        line2.set_ydata(diff_class_normal[:,j,i].flatten())
        line3.set_ydata(diff_book_normal[:,j,i].flatten())

        fig.canvas.draw_idle()


    epsilon_slider.on_changed(update)
    delta_slider.on_changed(update)

    ax_reset = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(ax_reset, 'Reset', hovercolor='0.975')

    def reset(event):
        epsilon_slider.reset()
        delta_slider.reset()


    button.on_clicked(reset)
    plt.show()

    ##########################################################################################################
    # PART 5: GO FOR GOLD (TWICE!)
    ##########################################################################################################

    # given constraints:
    d = 10
    epsilon = np.array([0.1])
    delta = np.array([0.05])

    # get the tighter bound (class bound)
    bound_class = class_bound(epsilon, delta, np.array([d]))

    # gets maximum difference (only need 1 point to be correct)
    emp_n_s_negative = get_empirical_bounds(d, T, epsilon, delta, bound_class, sampler=UniformSampler, positive='none')
    # gets minimum difference (needs MANY many points to be correct)
    emp_n_s_positive = get_empirical_bounds(d, T, epsilon, delta, bound_class, sampler=UniformSampler, positive='all')

    diff_negative = np.abs(bound_class-emp_n_s_negative).item()
    print('Difference between bounds when D: X = [-1, 1]^d, Y = -1: ', diff_negative)
    diff_positive = np.abs(bound_class-emp_n_s_positive).item()
    print('Difference between bounds when D: X = [-1, 1]^d, Y = +1: ', diff_positive)

    ##########################################################################################################
    # PART 6: JUST FOR FUNSIES
    ##########################################################################################################

    # z in open interval (0, 1)
    # My solution for Part 5 works for Part 6 :).


################################################################################################################
###################                                  - Soyon <3                              ###################
################################################################################################################
