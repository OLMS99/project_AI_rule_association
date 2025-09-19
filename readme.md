# Algoritmos de associação de regra

Este repositório foi feito junto com o trabalho de conclusão de curso de Ciência da Computação na Pontifícia Universidade Católica do Rio de Janeiro (PUC-Rio)
Escrito em português por Olavo Lucas Manhães de Souza e orientado por Augusto Baffa
o artigo está disponível para leitura aqui: [link ainda a ser providenciado]

# Uso

o projeto ainda vai ser feito para funcionar como um python package e ser compativel com modelos de bibliotecas populares como sklearn e tensorflow

## planejamento futuro

fora o que foi explicitado anteriormente, ainda há como objetivos:

- implementar outros algoritmos na biblioteca
- oferecer instruções, o artigo e logs em outras linguas, focando de inicio em inglês, francês e espanhol
- automatizar e padronizar testes

# estrutura de uma regra no projeto

uma regra é entendida como um nó com um critério e dois filhos, esquerdo quando o resultado é falso e direito quando o resultado é verdadeiro
esse nó é modular, permitindo fazer critérios mais complexos, criando uma arvore de nós

o critério normalmente é comparando um campo de entrada, especificado por coordenadas especificadas pelo nó, com um valor

no projeto existe outras estruturas de regras, regra MofN que é utilizada pelo algoritmos MofN, e outras que ainda não foram testadas completamente
sendo mais ideias elaboradas que não são utilizadas pelos algoritmos por agora

# Explicação resumida dos algoritmos:

## algoritmo Knowledge trace (KT)

Algoritmo simples, ótimo para iniciantes aprenderem sobre associação de regra

O algoritmos forma regras sobre cada neurônio na rede a partir de suas ligações para a camada seguinte

AVISO:  este algoritmo está incompleto 
        ele demora exponencialmente mais quanto maior a rede analisada em termos de ligações

## algoritmo Majority of N (MofN)

Algoritmo que lida com a estrutura de regra de mesmo nome, a regra resulta em verdadeiro se M of mais critérios de um total de N são verdadeiros

AVISO:  este algoritmo está incompleto

## algoritmo Rule Extraction as Learning (REL)

Este algoritmo analisa as entradas e as saídas da rede, uma abordagem pedagógica, diferente dos anteriores, que tem uma abordagem de decomposição

AVISO:  este algoritmo está incompleto

## algoritmo Relation-Extraction Neural(RxREN)

Este algoritmo analisa tanto as entradas e as saídas da rede quanto seus componentes, uma abordagem eclética
