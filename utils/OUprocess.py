import numpy as np
import matplotlib.pyplot as plt


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x_prev = x0

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


def main():
    ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(2), x0=[0.2, 0.5])
    plt.figure('data')
    y = []
    t = np.linspace(0, 100, 10000)
    for _ in t:
        y.append(ou_noise())
    # plt.plot(t, y)
    # plt.show()


if __name__ == "__main__":
    main()
