# Curso de Keras con TensorFlow

Las redes neuronales artificiales son un intento por simular el comportamiento de aprendizaje del cerebro. Los primeros experimentos se bañan en conectar partes del cerebro para estimular el aprendizaje

[https://images.app.goo.gl/aJ2Jga1WPVdSqxXv7](https://images.app.goo.gl/aJ2Jga1WPVdSqxXv7)
![https://images.app.goo.gl/aJ2Jga1WPVdSqxXv7](https://magiquo.com/wp-content/uploads/2019/11/neurona.png)

### Librerías disponibles

1. [Tensorflow](https://www.tensorflow.org/)
2. [PyTorch](https://pytorch.org/)

Inteligencia artificial ⇒ Busca replicar inteligencia humana.

Machine learning ⇒ Técnicas que busca replicar el aprendizaje automático.

Deep learning ⇒ Aprendizaje profundo, 

| Machine learning | Deep learning |
| --- | --- |
| Implementa lógica de negocio | Solo red neuronal |
|  | Peligro con el overfitting, (sobre entrenar) |

## Neuronas

    Células nerviosas interconectadas

    El perceptron, es una neurona artificial que busca imitar el funcionamiento de las neuronas del cerebro, el objetivo es tener varios perceptores con el fin de comunicarlos entre si, esto se hace con el fin de incluir casos atípicos?, se usa en el aprendizaje supervisado, de esta manera va alterando los pesos a medida que los resultados van coincidiendo con los datos reales

![Screenshot 2023-08-18 at 9.03.11 PM.png](Curso%20de%20Keras%20con%20TensorFlow%20301e3a5867174c7c8a4a06bdc1830c9b/Screenshot_2023-08-18_at_9.03.11_PM.png)

## Arquitectura

Sea n el numero de  entradas  y m en numero de neuronas (perceptrones) 

$$
\begin{vmatrix}
W_{11} & W_{12} & W_{13} & ... & W_{1n} \\
W_{21} & W_{22} & W_{23} & ... & W_{2n} \\                              ... & ... & ... &  & ... \\                                             W_{m1} & W_{m2} & W_{m3} & ... & W_{mn} \\
\end{vmatrix}_{mxn} *           \begin{vmatrix}
X_{1} \\
X_{2} \\                              X_{3} \\                                         ...   \\                                                       X_{n} \\
\end{vmatrix}_{n x 1} = \begin{vmatrix}
X1*W_{11} + X_2 *W_{12} + X_3* W_{13} + ...  + X_n *W_{1n} \\
X1*W_{21} + X_2 *W_{22} + X_3* W_{23} + ...  + X_n *W_{2n} \\                    ...  \\                          X1*W_{m1} + X_2 *W_{m2} + X_3* W_{m3} + ...  + X_n *W_{mn}\\
\end{vmatrix}_{mx1}  
$$

## Funciones de activación

zona a posterior de una neurona en donde existe un filtro, función de activación , que modifica el valor resultado , transmite la información generada por la los pesos y las entradas, esto es util para resolver problemas con dificultad alta.

```python
import numpy as np

sigmoid = lambda x : 1 / (1 + np.exp(-x))
step = lambda x: np.piecewise(x,[x<0, x>=0], [0, 1])
relu = lambda x : np.piecewise(x,[x<0, x>=0], [0, lambda a : a])
```

## Funciones de perdida

Son funciones que miden el porcentaje de error que tuvo el modelo de redes, para efectos prácticos el valor mas bajo es el mejor

- Learning rate ⇒ Quiere decir el paso con el que va a ir midiendo el valor. Considere cls siguiente caso
    - Pasos muy cortos, mayor capacidad para encontrar el mínimo error, no obstante tiene un coste computacional elevado.
    - Pasos muy largos, rendimiento pésimo, puede no encontrar el valor mínimo  de la función perdida debido a su alta volatilidad en la función

Momento ⇒ Se usa para calcular la tasa o la velocidad de Learning rate con la finalidad de encontrar el mínimo global.

### Tensores, tipos de datos

- Escalar, no tiene dimension.
- Vector, una dimension.
- Matrix, dos vectores.
- Tensor, mayor a 3 dimensiones.

Series de tiempo, 3 dimensiones, 

1. Cantidad de ejemplos.
2. Characteristics de los ejemplos
3. Cambio del ejemplo en el tiempo.

4 dimensiones

Imágenes, rgb, + cantidades de ejemplos 

## Manejo de  datos

|Train | Test | Val|

Durante el entrenamiento Val y Train

Test, se realiza de manera ética con el fin de validar que tan efectiva es el entrenamiento

Referencias 

[https://www.famaf.unc.edu.ar/~revm/digital24-3/redes.pdf](https://www.famaf.unc.edu.ar/~revm/digital24-3/redes.pdf)

[https://www.famaf.unc.edu.ar/~revm/digital24-3/redes.pdf](https://www.famaf.unc.edu.ar/~revm/digital24-3/redes.pdf)

[https://datascientest.com/es/perceptron-que-es-y-para-que-sirve#:~:text=Un perceptrón es una neurona,redes neuronales del Deep Learning](https://datascientest.com/es/perceptron-que-es-y-para-que-sirve#:~:text=Un%20perceptr%C3%B3n%20es%20una%20neurona,redes%20neuronales%20del%20Deep%20Learning).