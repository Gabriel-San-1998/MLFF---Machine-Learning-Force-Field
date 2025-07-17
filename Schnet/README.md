# SchNet

A biblioteca SchNet utiliza dados de força e/ou energia para prever propriedades de sistemas de interesse.

Realizamos dois tipos de estudo até o momento:

1. **Treinamento com forças, energias e geometrias**:  
   Um teste utilizando forças, geometrias e energias para treinar uma rede neural capaz de aprender o comportamento do sistema e realizar minimização de energia e simulações de dinâmica molecular, seja diretamente via ASE ou de forma indireta com o LAMMPS.  
   O algoritmo utilizado funciona, mas apresenta resultados insatisfatórios, possivelmente devido à baixa quantidade de dados (apenas 6 estruturas de treinamento).

2. **Treinamento apenas com energia e geometrias**:  
   Um teste utilizando apenas geometrias e energias. Usando um banco de dados com mais de 100 estruturas, conseguimos treinar uma rede capaz de prever a energia total de estruturas do mesmo tipo. Atualmente, estamos tentando obter as energias de interação entre pares para executar simulações de dinâmica molecular via LAMMPS.

Como destacado no artigo *"SchNet: A continuous-filter convolutional neural network for modeling quantum interactions"*, o uso combinado de forças e energias durante o treinamento tende a fornecer resultados mais precisos do que o uso exclusivo de energias.

