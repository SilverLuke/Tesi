# Tesi Argentieri Luca

La tesi consiste nello studio di modifiche architetturali di modelli di tipo ESN.

Modelli usati:
- ESN1: ESN di tipo standard
- ESN2: Il reservoir è composto da N sub-reservoir ognuno con dimensione uguale a UNITS / N, non comunicanti tra di loro
- ESN3: Il reservoir è composto da N sub-reservoir ognuno con dimensione uguale a UNITS / N, tra di loro connessi
- ESN4: Il reservoir è composto da N sub-reservoir ognuno con dimensione variable, tra di loro connessi

# Risultati ottenuti


Per maggiori informazioni leggere [qui](DESCRIPTION.md).

# Try it yourself

I dataset sono stati presi da [qui](http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip).
```bash
git clone https://github.com/SilverLuke/Tesi.git
cd Tesi
python -m venv venv
source venv/bin/acitivate # Se usi fish -> source venv/bin/acitivate.fish
pip install -r requirements.txt
cd datasets
wget http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip
unzip Multivariate2018_ts.zip
rm Multivariate2018_ts.zip
```

Per eseguire i modelli il notebook si trova in ```notebooks/main.ipynb```