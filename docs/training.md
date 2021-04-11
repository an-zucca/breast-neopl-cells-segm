##### Addestramento
Il modello di segmentazione delle istanze utilizzato è Mask R-CNN, nella sua implementazione originale resa disponibile all'interno della piattaforma [Detectron2](https://github.com/facebookresearch/detectron2) \[4\]

Come backbone del modello si è utilizzata la ResNet-101 \[5\].

Le immagini prima di entrare nel batch sono state soggette a data augmentation: riflesse (in orizzontale o verticale), 
ruotate (di 90, 180 o 270 gradi), modificate in contrasto (± 20%), saturazione (± 20%) e luminosità (± 20%) in modo casuale.  

Per l'ottimizzazione si è utilizzata la discesa stocastica del gradiente con momentum di 0.9.
 
Il tasso di apprendimento (learning rate) è stato inizializzato a 0.001 dopo 1000 iterazioni di warm up, e diviso per 10 all'iterazione 2000 e 2600. 

Il decadimento del peso (weight decay) è stato impostato a 0.0001.

Si è interrotto l'addestramento all'iterazione 3000.