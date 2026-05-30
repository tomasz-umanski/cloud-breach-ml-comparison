# Porównanie skuteczności metod uczenia głębokiego i lasów losowych w prognozowaniu naruszeń danych w chmurze

Projekt porównujący empirycznie skuteczność i wydajność klasycznych
lasów losowych oraz sieci głębokich (MLP, 1D-CNN, LSTM) w zadaniu wykrywania
(prognozowania) naruszeń danych w chmurze, sformułowanym jako binarna
klasyfikacja przepływów sieciowych (ruch normalny vs. atak).

Repozytorium zawiera:

* **`src/`** - pełny, uruchamialny pipeline w Pythonie (dane -> preprocessing ->
  trening -> ewaluacja -> wykresy i tabele),
* **`report/`** - sprawozdanie w LaTeX (15-20 stron) wraz z wygenerowanymi
  rysunkami i tabelami,
* **`results/`** - artefakty eksperymentu (metryki w JSON, wykresy PNG).

## Dane

Zbiorem docelowym jest **CSE-CIC-IDS2018** - ruch sieciowy zebrany na
infrastrukturze chmurowej AWS przez Canadian Institute for Cybersecurity,
zawierający ataki typu brute force, DoS/DDoS, ataki webowe, infiltrację oraz
botnet. Pełny zbiór wymaga ręcznego pobrania (kilka GB), dlatego:

* jeśli w katalogu `data/raw/` znajdą się pliki CSV CIC-IDS2018, zostaną one
  automatycznie wczytane (podzbiór do ~200 tys. wierszy),
* w przeciwnym razie uruchamiany jest **generator danych syntetycznych** o
  charakterystyce zbliżonej do cech przepływów CIC-IDS, co zapewnia pełną
  odtwarzalność eksperymentu offline.

Źródło danych ustawia się w `src/config.py` (`DATA_SOURCE`) lub flagą
`--source`.

## Instalacja

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Uwaga: modele głębokie zaimplementowano w **PyTorch** (TensorFlow nie posiada
> jeszcze wheela dla Pythona 3.14).

## Uruchomienie eksperymentu

```bash
# Pełny eksperyment (domyślnie: auto -> dane realne lub syntetyczne)
python -m src.run_all

# Wymuszenie danych syntetycznych
python -m src.run_all --source synthetic

# Szybki przebieg kontrolny
python -m src.run_all --source synthetic --n-samples 20000 --epochs 10
```

Po zakończeniu w `results/` pojawiają się metryki i wykresy, a w
`report/generated/` tabele i makra LaTeX wykorzystywane przez sprawozdanie.

## Kompilacja sprawozdania

```bash
cd report
pdflatex main.tex && pdflatex main.tex   # dwukrotnie, dla spisu treści i odnośników
```

## Struktura projektu

```
src/
  config.py            # konfiguracja eksperymentu
  data/                # generator, pobieranie, preprocessing
  models/              # classical.py (RF, LogReg), deep.py (MLP/CNN/LSTM)
  evaluate.py          # metryki
  train.py             # pętla treningowo-ewaluacyjna
  plots.py             # generowanie rysunków
  tables.py            # eksport tabel i makr LaTeX
  run_all.py           # punkt wejścia pipeline'u
report/                # sprawozdanie LaTeX
results/               # metryki i rysunki (generowane)
```
