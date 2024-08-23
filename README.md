# Projeto de Transfer Learning com ResNet50
### Objetivo
Este projeto foi desenvolvido para fins educativos, com o objetivo de explorar o uso de Transfer Learning utilizando a arquitetura ResNet50 para classificar imagens em diferentes categorias de lixo. É importante notar que este projeto foi realizado com um número limitado de amostras de treinamento, o que impacta os resultados e a precisão do modelo.

### Dataset
O dataset utilizado neste projeto foi o TrashNet, composto por imagens de diferentes tipos de lixo (papel, plástico, metal, vidro, etc.). Para este experimento, foram utilizadas 100 imagens para cada classe, totalizando 600 imagens. Esse número é relativamente pequeno, o que pode limitar a capacidade do modelo de generalizar para novas imagens.

### Treinamento do Modelo
A arquitetura ResNet50 pré-treinada na ImageNet foi utilizada como base para este modelo. As últimas camadas foram ajustadas para se adequar ao problema de classificação de 6 classes diferentes. O modelo foi treinado por 6 épocas, utilizando um conjunto de dados de treino e outro de validação.

#### Desafios
Durante o treinamento, foi observado que em algumas épocas os valores de accuracy e loss zeraram, indicando que o treinamento não avançou como esperado nessas épocas. Isso pode estar relacionado à quantidade limitada de dados de treinamento ou à necessidade de ajustar o número de épocas e outros hiperparâmetros para melhorar o desempenho do modelo.

#### Resultados
Os gráficos gerados a partir do histórico de treinamento mostram a precisão e a perda durante o treinamento e validação:

 - Training and Validation Accuracy: Este gráfico mostra a variação da precisão do modelo ao longo das épocas. É possível observar flutuações significativas, indicando que o modelo não convergiu de forma consistente.

 - Training and Validation Loss: O gráfico de perda mostra uma tendência de queda, mas também com flutuações, refletindo a dificuldade do modelo em ajustar-se aos dados limitados.

![foo](https://github.com/user-attachments/assets/8c188503-c026-4a55-b17a-9e81ddf7cec9)


### Teste do Modelo
Após o treinamento, o modelo foi testado com uma imagem de papel amassado. O modelo classificou esta imagem como metal com uma probabilidade de 34%, enquanto atribuiu 28% de probabilidade à classe papel.

![image](https://github.com/user-attachments/assets/7a07d90a-d666-4b9b-b85e-2e070da8b9b5)

Este resultado pode ser considerado aceitável, dado que o modelo teve acesso a apenas 100 imagens por classe para o treinamento. A proximidade das probabilidades entre as classes metal e papel sugere que o modelo foi capaz de captar algumas características compartilhadas entre os dois, mas a quantidade limitada de dados de treinamento pode ter dificultado a precisão da classificação.

### Conclusão e Próximos Passos
Este projeto ilustra a aplicabilidade do Transfer Learning com a ResNet50 para problemas de classificação de imagens, mesmo com um dataset pequeno. Para melhorar os resultados, recomenda-se:

1. Aumentar o número de imagens de treinamento: Um conjunto de dados maior ajudaria o modelo a generalizar melhor e reduziria a flutuação nos resultados.
2. Investigar as épocas com valores zerados: Analisar por que algumas épocas tiveram valores de accuracy e loss zerados e ajustar os hiperparâmetros conforme necessário.
3. Ajustar o número de épocas: O modelo pode se beneficiar de um maior número de épocas, ou de uma política de aprendizado mais sofisticada, como o ajuste da taxa de aprendizado.
Você pode atuar nesse projeto, é só criar uma nova branch que vou avaliar todas as contribuições.
