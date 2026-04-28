## Zbiory danych (Datasets)

* **Dataset główny:** [Movie Lens 25M](https://grouplens.org/datasets/movielens/25m/)

### Przygotowanie danych
Dane należy wypakować do poniższej struktury katalogów:
```text
├── data/
│   ├── 01_raw/# Dane źródłowe
│   │   ├── links.csv
│   │   ├── movies.csv
│   │   └── ratings.csv
```
### Ustawienie środowiska
1. Instalacja zależności Python (Conda)

```Bash
conda env create -f environment.yml
conda activate recommender-ml
```
## Obsługa Kedro
Aby uruchomić wybrany potok: 
```bash
kedro run --pipeline NAZWA_POTOKU
```
### Dostępne potoki:
* preprocess_movies – przygotowanie danych filmów, one hot encoding dla gatunków, poprawienie numerowania ID
* preprocess_users - przygotowanie danych historii oglądania
* train_baseline - trenowanie modelu bazowego
* train_prod - trenowanie modelu produkcyjnego, bardziej złożonego
* test - tesowanie modeli, obliczenie metryk
* export - eksportowanie modelu produkcyjnego do ONNX, eksportowanie wektorów cech do pliku csv

QUICKSTART
```bash
kedro run --pipeline preprocess_movies
kedro run --pipeline preprocess_users
kedro run --pipeline train_baseline
kedro run --pipeline train_prod
kedro run --pipeline test
kedro run --pipeline export