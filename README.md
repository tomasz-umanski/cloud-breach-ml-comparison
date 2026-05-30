# Porównanie skuteczności metod uczenia głębokiego i lasów losowych w prognozowaniu naruszeń danych w chmurze

Empiryczne porównanie klasycznych **lasów losowych** oraz sieci głębokich (**MLP**, **CNN 1D**, **LSTM**) w zadaniu wykrywania (prognozowania) naruszeń danych w chmurze. Problem sformułowano jako binarną klasyfikację przepływów sieciowych: ruch normalny vs. atak.

Repozytorium zawiera:

* **`src/`** — pipeline w Pythonie (dane → preprocessing → trening → ewaluacja → wykresy),
* **`Sprawozdanie.docx`** — główne sprawozdanie (Word, styl wzorca BIHSO), generowane skryptem,
* **`report/`** — alternatywne sprawozdanie w LaTeX (PDF),
* **`results/`** — metryki i wykresy (generowane; katalog gitignored).

## Porównywane modele

| Model | Biblioteka |
|---|---|
| Regresja logistyczna (baseline) | scikit-learn |
| Las losowy | scikit-learn |
| MLP, CNN 1D, LSTM | PyTorch |

## Dane

Zbiorem docelowym jest **CSE-CIC-IDS2018** — ruch sieciowy zebrany na infrastrukturze chmurowej AWS (Canadian Institute for Cybersecurity): brute force, DoS/DDoS, ataki webowe, infiltracja, botnet itd.

| Tryb | Opis |
|---|---|
| **`auto`** (domyślny) | Wczytuje pliki CSV z `data/raw/`, jeśli są dostępne; w przeciwnym razie używa generatora syntetycznego |
| **`cicids`** | Wymaga plików CSE-CIC-IDS2018 w `data/raw/` |
| **`synthetic`** | Zawsze generator danych syntetycznych (offline, szybki) |

Pełny zbiór ma kilka GB. Program wczytuje go **strumieniowo** (chunkami), losowo próbkuje porcję z każdego pliku dziennego i zapisuje wycinek w cache (`data/processed/cicids_subset_120000.csv`), żeby kolejne uruchomienia nie czytały gigabajtów od nowa.

Pliki CSV umieść w:

```
data/raw/
  02-14-2018.csv
  02-15-2018.csv
  ...
```

## Instalacja

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Modele głębokie zaimplementowano w **PyTorch** (brak wheela TensorFlow dla Pythona 3.14). Do generowania sprawozdania Word potrzebny jest pakiet **python-docx**.

## Uruchomienie

### 1. Eksperyment

```bash
# Pełny eksperyment (auto: dane realne lub syntetyczne)
python -m src.run_all

# Wymuszenie danych syntetycznych
python -m src.run_all --source synthetic

# Szybki przebieg kontrolny
python -m src.run_all --source synthetic --n-samples 20000 --epochs 10
```

Po zakończeniu:

* `results/metrics/` — metryki w JSON,
* `results/figures/` — wykresy PNG,
* `report/figures/` — kopie wykresów do sprawozdania,
* `report/generated/` — tabele i makra LaTeX (opcjonalnie).

### 2. Sprawozdanie Word (.docx)

W katalogu głównym musi leżeć szablon **`Wzór.docx`** (strona tytułowa i style). Następnie:

```bash
python -m src.build_docx
```

Powstaje plik **`Sprawozdanie.docx`** z treścią, wykresami i tabelami zsynchronizowanymi z wynikami eksperymentu.

### 3. Sprawozdanie LaTeX (opcjonalnie)

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Przykładowe wyniki (CSE-CIC-IDS2018, wycinek ~115 tys. połączeń)

| Model | F1 | ROC-AUC | Czas treningu |
|---|---|---|---|
| CNN 1D | 0,913 | 0,969 | ~172 s |
| Las losowy | 0,912 | 0,968 | ~4 s |
| MLP | 0,898 | 0,967 | ~12 s |
| LSTM | 0,882 | 0,943 | ~487 s |
| Regresja logistyczna | 0,783 | 0,945 | ~1 s |

Las losowy i najlepsza sieć osiągają praktycznie identyczną skuteczność; las losowy uczy się jednak kilkadziesiąt razy szybciej.

## Struktura projektu

```
.
├── BIHSO Projekt.docx     # szablon strony tytułowej i stylów (wymagany do build_docx)
├── Sprawozdanie.docx      # wygenerowane sprawozdanie (po python -m src.build_docx)
├── data/
│   ├── raw/               # pliki CSV CSE-CIC-IDS2018 (gitignored)
│   └── processed/         # cache wycinka (gitignored)
├── report/                # sprawozdanie LaTeX + figures/
├── results/               # metryki i wykresy (gitignored, generowane)
├── src/
│   ├── config.py          # parametry eksperymentu
│   ├── run_all.py         # uruchomienie całego potoku
│   ├── build_docx.py      # generowanie Sprawozdanie.docx
│   ├── docx_builder.py    # helpery Word (style BIHSO)
│   ├── data/              # loader, download, preprocess, synthetic
│   ├── models/            # classical.py, deep.py
│   ├── train.py           # pętla treningowo-ewaluacyjna
│   ├── evaluate.py        # metryki
│   ├── plots.py           # wykresy
│   └── tables.py          # eksport tabel LaTeX
└── requirements.txt
```

## Konfiguracja

Główne parametry w [`src/config.py`](src/config.py):

* `DATA_SOURCE` — `auto` / `cicids` / `synthetic`
* `SEED` — ziarno losowości (powtarzalność)
* `RandomForestConfig`, `DeepConfig` — hiperparametry modeli

Nadpisanie z linii poleceń: `--source`, `--epochs`, `--n-samples`, `--seed`.

## Licencja i dane zewnętrzne

Zbiór CSE-CIC-IDS2018 udostępniany jest przez [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2018.html) — wymaga akceptacji warunków i ręcznego pobrania.
