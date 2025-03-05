# Abordagem inicial

- Nessa fase inicial, os patches foram feitos pegando imagem de 64x64 em sequencia, ou seja, nao era em pontos aleatorios. 

- Por nao ser de forma aleatoria, cada classe tinha uma qtd limitada de amostras e para contornar isso foi feito DA. Assim todas as classes foram jogadas para 4000 samples 
para ficar balanceado. 

- O classficador composto por uma rede simples 

- A rescontrucao da imagem também foi de forma sequencia, as novas imagens era cortadas em 64x64, classificadas e recosntruidas em sequencia 
