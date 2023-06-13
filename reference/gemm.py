import numpy as np


def quantization(x, s, z, alpha_q, beta_q):

    x_q = np.round(1 / s * x + z, decimals=0)
    x_q = np.clip(x_q, a_min=alpha_q, a_max=beta_q)

    return x_q


def quantization_int8(x, s, z):

    x_q = quantization(x, s, z, alpha_q=0, beta_q=255)
    x_q = x_q.astype(np.uint8)

    return x_q


def dequantization(x_q, s, z):

    # x_q - z might go outside the quantization range.
    x_q = x_q.astype(np.int32)
    x = s * (x_q - z)
    x = x.astype(np.float32)

    return x


def generate_quantization_constants(alpha, beta, alpha_q, beta_q):

    # Affine quantization mapping
    s = (beta - alpha) / (beta_q - alpha_q)
    z = int((beta * alpha_q - alpha * beta_q) / (beta - alpha))

    return s, z


def generate_quantization_int8_constants(alpha, beta):

    b = 8
    alpha_q = 0
    beta_q = 255

    s, z = generate_quantization_constants(alpha=alpha,
                                           beta=beta,
                                           alpha_q=alpha_q,
                                           beta_q=beta_q)

    return s, z


def quantization_matrix_multiplication_int8(X_q, W_q, b_q, s_X, z_X, s_W, z_W,
                                            s_b, z_b, s_Y, z_Y):

    p = W_q.shape[0]

    # Y_q_simulated is FP32
    Y_q_simulated = (z_Y + (s_b / s_Y * (b_q.astype(np.int32) - z_b)) + (
        (s_X * s_W / s_Y) *
        (np.matmul(X_q.astype(np.int32), W_q.astype(np.int32)) -
         z_W * np.sum(X_q.astype(np.int32), axis=1, keepdims=True) - z_X *
         np.sum(W_q.astype(np.int32), axis=0, keepdims=True) + p * z_X * z_W)))

    Y_q_simulated = np.round(Y_q_simulated, decimals=0)
    Y_q_simulated = np.clip(Y_q_simulated, a_min=0, a_max=255)
    Y_q_simulated = Y_q_simulated.astype(np.uint8)

    return Y_q_simulated


def main():

    # Set random seed for reproducibility
    #random_seed = 0
    #np.random.seed(random_seed)

    # Random matrices
    m = 1
    p = 1
    n = 1

    # X
    alpha_X = -100.0
    beta_X = 80.0
    s_X, z_X = generate_quantization_int8_constants(alpha=alpha_X, beta=beta_X)
    X = np.random.uniform(low=alpha_X, high=beta_X,
                          size=(m, p)).astype(np.float32)
    X_q = quantization_int8(x=X, s=s_X, z=z_X)
    X_q_dq = dequantization(x_q=X_q, s=s_X, z=z_X)

    # W
    alpha_W = -20.0
    beta_W = 10.0
    s_W, z_W = generate_quantization_int8_constants(alpha=alpha_W, beta=beta_W)
    W = np.random.uniform(low=alpha_W, high=beta_W,
                          size=(p, n)).astype(np.float32)
    W_q = quantization_int8(x=W, s=s_W, z=z_W)
    W_q_dq = dequantization(x_q=W_q, s=s_W, z=z_W)

    # b
    alpha_b = -500.0
    beta_b = 500.0
    s_b, z_b = generate_quantization_int8_constants(alpha=alpha_b, beta=beta_b)
    b = np.zeros((1, n)).astype(np.float32)
    b_q = quantization_int8(x=b, s=s_b, z=z_b)
    b_q_dq = dequantization(x_q=b_q, s=s_b, z=z_b)

    # Y
    alpha_Y = -3000.0
    beta_Y = 3000.0
    s_Y, z_Y = generate_quantization_int8_constants(alpha=alpha_Y, beta=beta_Y)
    Y_expected = np.matmul(X, W) + b
    Y_q_expected = quantization_int8(x=Y_expected, s=s_Y, z=z_Y)

    Y_expected_prime = np.matmul(X_q_dq, W_q_dq) + b_q_dq
    Y_expected_prime_q = quantization_int8(x=Y_expected_prime, s=s_Y, z=z_Y)
    Y_expected_prime_q_dq = dequantization(x_q=Y_expected_prime_q,
                                           s=s_Y,
                                           z=z_Y)

    print("Expected FP32 Y:")
    print(Y_expected)
    print("Expected FP32 Y Quantized:")
    print(Y_q_expected)

    Y_q_simulated = quantization_matrix_multiplication_int8(X_q=X_q,
                                                            W_q=W_q,
                                                            b_q=b_q,
                                                            s_X=s_X,
                                                            z_X=z_X,
                                                            s_W=s_W,
                                                            z_W=z_W,
                                                            s_b=s_b,
                                                            z_b=z_b,
                                                            s_Y=s_Y,
                                                            z_Y=z_Y)
    Y_simulated = dequantization(x_q=Y_q_simulated, s=s_Y, z=z_Y)

    print("Expected Quantized Y_q from Quantized Matrix Multiplication:")
    print(Y_q_simulated)
    print(
        "Expected Quantized Y_q from Quantized Matrix Multiplication Dequantized:"
    )
    print(Y_simulated)

    # Ensure the algorithm implementation is correct
    assert (np.array_equal(Y_simulated, Y_expected_prime_q_dq))
    assert (np.array_equal(Y_q_simulated, Y_expected_prime_q))


if __name__ == "__main__":

    main()
