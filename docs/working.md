##### Funzionamento

A ogni accesso, carica un'immagine di tessuto mammario, tra quelle disponibili in-app.

Per ogni cellula nel tessuto, riporta: bounding box, maschera di segmentazione e classe ground truth.
Restituisce inoltre le predizioni da un file json che memorizza gli output del modello in una variante del [formato COCO] (https://cocodataset.org/#format-data).

Consente di modificare la soglia di confidenza per filtrare le predizioni.

Riporta la distribuzione delle cellule neoplastiche e non neoplastiche nel tessuto.

Calcola caratteristiche morfologiche di base delle cellule: area, perimetro, circolarità, solidità.
_____________________________________________________________________________________________________________________________________________________