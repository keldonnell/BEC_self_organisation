import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(amplitudes, num_points=2000):
    """
    Returns arrays for <sqrt(S)>, sqrt(<S>), and their difference for S(x)=A(sin x + 1).
    """
    x = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    mean_sqrt = []
    sqrt_mean = []
    diff = []

    for A in amplitudes:
        s = A * (np.sin(x)) + 1.0
        m_sqrt = np.mean(np.sqrt(s))
        m = np.mean(s)
        mean_sqrt.append(m_sqrt)
        sqrt_mean.append(np.sqrt(m))
        diff.append(m_sqrt - np.sqrt(m))

    return np.array(mean_sqrt), np.array(sqrt_mean), np.array(diff)


def main():
    amplitudes = np.logspace(0, -8, 200)  # 1 down to 1e-8
    mean_sqrt, sqrt_mean, diff = compute_metrics(amplitudes)

    fig, ax = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax[0].plot(amplitudes, mean_sqrt, label=r"$\langle \sqrt{S} \rangle$")
    ax[0].plot(amplitudes, sqrt_mean, label=r"$\sqrt{\langle S \rangle}$", linestyle="--")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("Value")
    ax[0].legend()
    ax[0].grid(True, linestyle=":", alpha=0.5)

    ax[1].plot(amplitudes, diff, color="black")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Amplitude A in S(x)=A(sin x + 1)")
    ax[1].set_ylabel(r"$\langle \sqrt{S} \rangle - \sqrt{\langle S \rangle}$")
    ax[1].grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
