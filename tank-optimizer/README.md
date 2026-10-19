# Otimizador de Tanque

Este projeto é uma aplicação web desenvolvida em Flask para otimizar o design de um tanque cilíndrico, minimizando seu custo de fabricação com base em restrições de volume e dimensões. A aplicação permite ao usuário comparar visualmente a eficiência de diferentes algoritmos de otimização não linear.

## Estrutura do Projeto

O projeto segue uma estrutura baseada no padrão Model-View-Controller (MVC) para organizar o código:

-   **/controllers**: Contém a lógica que conecta a interface do usuário aos modelos de otimização.
-   **/models**: Inclui a definição do problema (função de custo, restrições) e a implementação dos algoritmos de otimização (Steepest Descent, Newton, DFP).
-   **/static**: Armazena os arquivos estáticos (JavaScript, CSS, imagens). O `main.js` é responsável pela interatividade da página, chamadas de API e renderização dos gráficos com Plotly.
-   **/templates**: Contém o template `index.html` que estrutura a página web.
-   `app.py`: O ponto de entrada da aplicação Flask, que define as rotas da API.
-   `requirements.txt`: Lista as dependências do Python.
## Arquitetura da Interface (Frontend)

O fluxograma abaixo descreve a arquitetura da interface do usuário e o fluxo de interação, desde a entrada de dados até a visualização dos resultados.

```mermaid
graph TD
    Usuario[Usuário] --> UI{Interface Principal};

    subgraph UI [Interface Principal - index.html]
        direction LR
        subgraph "Entradas"
            PainelControle["Painel de Controle<br/>(Dados Manuais)"];
            Cenarios["Cenários de Teste<br/>(Dados Pré-configurados)"];
        end
        
        subgraph "Modos de Análise"
            Abas["Abas<br/>Individual / Comparativo"];
        end

        BotaoCalcular((Calcular));
    end

    PainelControle --> BotaoCalcular;
    Cenarios --> BotaoCalcular;

    BotaoCalcular -- Requisição p/ API<br/>de acordo com a aba ativa --> Backend[API Flask];

    subgraph "Visualização de Resultados"
      direction LR
      Backend -- Retorna Dados --> Individual["Análise Individual"];
      Backend -- Retorna Dados --> Comparativo["Modo Comparativo"];
    end

    subgraph Individual
        direction TB
        R1["Resultados Numéricos"];
        G1["Gráfico de Trajetória"];
        G2["Gráfico de Convergência"];
    end

    subgraph Comparativo
        direction TB
        Tabela["Tabela de Resultados"];
        G3["Gráfico Trajetórias<br/>Animado com Replay"];
        G4["Gráfico Convergência<br/>Animado com Replay"];
    end

    classDef card fill:#2c3e50,stroke:#bdc3c7,stroke-width:2px,color:#fff
    class UI,Individual,Comparativo card
```
## Como Executar

Para executar este projeto, primeiro clone o repositório e navegue até a pasta `optimzed_tank`.

```bash
git clone <URL_DO_REPOSITORIO>
cd optimzed_tank
```

A partir daqui, você pode configurar o ambiente de duas maneiras:

### Opção 1: Usando `venv` (Padrão Python)

1.  **Crie e Ative o Ambiente Virtual**

    -   No Linux/macOS:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    -   No Windows:
        ```bash
        python -m venv venv
        .\\venv\\Scripts\\activate
        ```

2.  **Instale as Dependências**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Execute a Aplicação**
    ```bash
    python3 app.py
    ```

### Opção 2: Usando `conda`

1.  **Crie e Ative o Ambiente Virtual**
    ```bash
    conda create --name optimized_tank python=3.12 -y
    conda activate optimized_tank
    ```

2.  **Instale as Dependências**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Execute a Aplicação**
    ```bash
    python3 app.py
    ```

Após executar um dos comandos acima, a aplicação estará disponível em `http://127.0.0.1:5000` no seu navegador.

