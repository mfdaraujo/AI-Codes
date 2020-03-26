import random
import numpy as np
import matplotlib.pyplot as plt

class algGenetico():
    
    ''' Classe para gerar a palavra informada,
        só pode ser utilizada para palavras com quantidade de letras pares, 
        o tratamento para crossover impar não foi implementado'''
        
    def __init__(self,palavra):
        
        '''Define o espaço de busca do alfabeto e o peso
        de cada letra da palavra para ser minimizado
        pelo algoritmo.'''
        
        self.espaco_busca = ['a','b','c','d','e','f','g','h', 
                             'i','j','k','l','m','n','o','p',
                             'q','r','s','t','u','v','w','x','y','z',' ']
        self.busca_tam = len(palavra)
        self.referencia = palavra
        self.populacao = []
        self.filho1 = []
        self.filho2 = []
        self.peso_referencia = []
        pesos_referencia = []
        
        for i in range(self.busca_tam):
            letra = self.referencia[i]
            peso_referencia = (self.espaco_busca.index(letra) + 1)
            pesos_referencia.append(peso_referencia)
            
        self.peso_referencia = pesos_referencia
        
    def fazerCromossomo(self):
        
        ''' Gera um cromossomo aleatório de mesmo 
            tamanho da palavra buscada'''
            
        letras = []
        self.cromossomo = ''
        espaco_busca_tam = len(self.espaco_busca)
        indices = np.random.randint(espaco_busca_tam , size = self.busca_tam)
        
        for i in indices:
        
            letras.append(self.espaco_busca[i])   
        
        letras = ''.join(letras)
        self.cromossomo = letras
        
        return self.cromossomo
    
    def fazerPopulacaoInicial(self,tamanho_populacao):
        
        ''' Cria a população com N cromossomos'''
        
        for _ in range(tamanho_populacao):
        
            self.populacao.append(self.fazerCromossomo())
        
        return True
    
    def adaptacao(self):
        
        ''' Mede o quão adaptado (soma da distância de 
        cada letra para a palavra original) os cromossomos
        da população atual são'''
        
        self.fitness = []
        populacao_tam = len(self.populacao)
        
        for pessoa in range(populacao_tam):
            
            palavra = self.populacao[pessoa]
            fitness_palavra = 0
            
            for i in range(self.busca_tam):
            
                letra = palavra[i]
                peso_palavra = (self.espaco_busca.index(letra) + 1)
                aux = abs(self.peso_referencia[i] - peso_palavra)
                fitness_palavra = fitness_palavra + aux
            
            self.fitness.append(fitness_palavra)
        
        return self.fitness
    
    def adaptadorGenetico(self, pai1, pai2):
        
        '''Realiza o crossover na metade do cromossomo,
           cada leitura do gene tem a chance de CHANCE_MUTAÇÃO
           de sofrer uma mutação e ser transcrito de forma diferente.
           Gera como resultado dois filhos com chance de mutação em cada gene'''
        
        self.filho1 = []
        self.filho2 = []
        espaco_busca_tam = len(self.espaco_busca)
        chance_mutacao = 0.05
        
        for gene in range(0,int(self.busca_tam/2)):
            
            self.filho1.append(pai2[gene+int(self.busca_tam/2)])
            
            if (random.random() <= chance_mutacao):
            
                letra = random.randint(0, espaco_busca_tam - 1)
                self.filho2.append(self.espaco_busca[letra])
            
            else:
                
                self.filho2.append(pai2[gene])
                
        for gene in range(int(self.busca_tam/2),int(self.busca_tam)):
            
            self.filho1.append(pai1[gene])
            
            if (random.random() <= chance_mutacao):
            
                letra = random.randint(0, espaco_busca_tam - 1)
                self.filho2.append(self.espaco_busca[letra])
            
            else:
                
                self.filho2.append(pai1[gene-int(self.busca_tam/2)])
        
        self.filho1 = ''.join(self.filho1)
        self.filho2 = ''.join(self.filho2)
        
        return self.filho1, self.filho2
    
    def novaPopulacao(self):
        
        '''Cria uma nova população mantendo o melhor elemento da 
            geração anterior, os pais dessa população nova é feita pelo
            método SELECAO'''
        
        melhor_candidato,_ = self.melhorCandidato()
        tamanho_populacao = len(self.populacao_pais)
        
        for i in range(int(tamanho_populacao/2)):
        
            self.adaptadorGenetico(self.populacao_pais[i*2], self.populacao_pais[i*2+1])
            self.populacao[i*2] = self.filho1
            self.populacao[i*2+1] = self.filho2
        
        self.populacao[i*2+1] = melhor_candidato
        
        return True
    
    def selecao(self, populacao_torneio):
        
        ''' Seleciona no conjunto população, através do método de torneio,
            os cromossomos que irão formar o conjunto de pais para aplicar
            os adaptadores genéticos de crossover e mutação'''
        
        self.populacao_pais = []
        tamanho_populacao = len(self.populacao)
        
        
        for n_torneios in range(tamanho_populacao):
            
            participantes_torneio = []
            pessoas = np.random.randint(tamanho_populacao, size = populacao_torneio)
            pessoas = [int(x) for x in pessoas]
            
            for pessoa in pessoas:
        
                participantes_torneio.append(self.fitness[pessoa]) 
                
            campeao_fitness = min(participantes_torneio)
            campeao_i = self.fitness.index(campeao_fitness)
            self.populacao_pais.append(self.populacao[campeao_i])
        
        return self.populacao_pais
        
    def melhorCandidato(self):
        
        '''Retorna o melhor candidato da população'''
        
        self.melhor_fitness = 0
        self.adaptacao()
        melhor_fitness = min(self.fitness)
        self.melhor_fitness = melhor_fitness
        melhor_fitness_indice = self.fitness.index(melhor_fitness)
        self.melhor_candidato = self.populacao[melhor_fitness_indice]
        
        return self.melhor_candidato,self.melhor_fitness

palavra = 'cade'

teste = algGenetico(palavra)
teste.fazerPopulacaoInicial(200)
teste.adaptacao()
teste.melhorCandidato()
geracao = 1
fitness = []

while (geracao <=100000):
    
    teste.selecao(5)
    teste.novaPopulacao()
    teste.adaptacao()
    teste.melhorCandidato()
    fitness.append(teste.melhor_fitness)
    print(geracao)
    print(teste.melhor_candidato)
    
    if (teste.melhor_fitness <= 0):
        break
    
    else:
        geracao +=1

solucao = teste.melhor_candidato
plt.plot(fitness)
