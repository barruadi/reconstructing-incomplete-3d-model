import matplotlib.pyplot as plt

def visualize_3d(original, missing, reconstructed):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(original[:, 0], original[:, 1], original[:, 2], c='blue', label='Original', alpha=0.6)
    ax.scatter(missing[:, 0], missing[:, 1], missing[:, 2], c='red', label='Missing', alpha=0.6)
    ax.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], c='green', label='Reconstructed', alpha=0.6)

    ax.legend()
    plt.show()

def visualize_preprocess(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', label='preprocess', alpha=0.6)

    ax.legend()
    plt.show()