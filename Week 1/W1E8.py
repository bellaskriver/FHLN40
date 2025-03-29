import matplotlib.pyplot as plt
import numpy as np

def plot_sine_wave():
    f = lambda x: np.sin(10 * x)
    x = np.linspace(0, 2, 100)
    plt.plot(x, f(x))
    plt.show()

def plot_sine_wave_with_labels():
    f = lambda x: np.sin(x)
    x = np.linspace(0, 2, 100)
    fig, ax = plt.subplots()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.plot(x, f(x), linestyle=':', color='r', label="$f(x)$")
    ax.legend()
    fig.tight_layout()
    plt.show()

def plot_contour():
    x = np.linspace(0, 2, 100)
    y = np.linspace(0, 1, 100)
    x, y = np.meshgrid(x, y, indexing='ij')
    g = lambda x, y: np.sin(2 * np.pi * x) * np.cos(np.pi * y)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("$f(x, y)$")
    cp = ax.contourf(x, y, g(x, y))
    fig.colorbar(cp)
    fig.tight_layout()
    plt.show()

def plot_pcolormesh():
    x = np.linspace(0, 2, 100)
    y = np.linspace(0, 1, 100)
    x, y = np.meshgrid(x, y, indexing='ij')
    g = lambda x, y: np.sin(2 * np.pi * x) * np.cos(np.pi * y)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("$f(x, y)$")
    cp = ax.pcolormesh(x, y, g(x, y))
    fig.colorbar(cp)
    fig.tight_layout()
    plt.show()

def plot_3d_surface():
    x = np.linspace(0, 2, 100)
    y = np.linspace(0, 1, 100)
    x, y = np.meshgrid(x, y, indexing='ij')
    g = lambda x, y: np.sin(2 * np.pi * x) * np.cos(np.pi * y)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    ax.set_title("$f(x)$")
    cp = ax.plot_surface(x, y, g(x, y), cmap=plt.cm.coolwarm, antialiased=False)
    fig.colorbar(cp, pad=0.2, ticks=np.linspace(-0.8, 0.8, 9))
    fig.tight_layout()
    plt.show()

def main():
    plot_sine_wave()
    plot_sine_wave_with_labels()
    plot_contour()
    plot_pcolormesh()
    plot_3d_surface()

main()