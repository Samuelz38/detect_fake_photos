import numpy as np

def calculate_edge_density(image : np.ndarray) -> float:

    """Calcula a densidade das bordas de uma imagem

    Parameters
    ----------
    image : numpy.ndarray
        Imagem para calcular a densidade das bordas
    
    Returns
    -------
    float
        Resultado do cálculo da densidade das bordas da imagem
    """

    return np.sum(image) / image.size