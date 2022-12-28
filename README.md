## Introduzione

MMFlow è un toolbox open source per il calcolo del flusso ottico basato su PyTorch ed è parte del progetto [OpenMMLab](https://openmmlab.com/). In questa fork, ho introdotto l'implementazione necessaria per supportare nuove pipeline in input al modello scelto, nuove tipologie di dataset e dichiarato differenti strategie di addestramento volte ad addestrare RAFT su dati sintetici e reali. 

## Pipeline 
La pipeline proposta utilizza un modello di rete neurale convoluzionale chiamato RAFT
che apprende la generazione del flusso ottico da dataset annotati forniti in input. Ricordiamo
che per annotazioni s’intende la disponibilità di dati con i rispettivi flussi ottici, utilizzabili
come tavole di verità. Tuttavia, come anticipato precedentemente, in contesti reali e dinamici
come uno scenario sottomarino, è molto complesso avere a disposizione dati annotati. Pertanto
nella pipeline è stato integrato il metodo di Depthstillation che consente di generare tavole di
verità verosimili, applicando trasformazioni geometriche alle immagini di input. Al termine
del processo di Depthstillation, si avranno a disposizione, quindi, i dati necessari per allenare
il modello utilizzato.

## Dati utilizzati 
I dataset utilizzati in fase di training sono un mix di dataset sintetici, ovvero generati dalla
computer vision ed un dataset sottomarino su cui è stata implementata la depthstillation.
Quest’ultimo è un dataset reale di nome CADDY.
- **FlyingChairs FyingThings3D**: si tratta di dataset sintetici di medie dimensioni,
il primo con 22872 immagini e l’ultimo con 25000 e relative annotazioni di flusso otti-
co. Sono rappresentate sedie ed oggetti comuni in primo piano su sfondo casuale che
compiono movimenti casuali planari. [54] A differenza di dataset come KITTI, che si
concentrano su task specifici, si apprezza la casualità e grandi moli di dati generati senza
ripetizioni o saturazioni. Addestramenti di reti condotti su dataset con queste caratte-
ristiche, dovrebbero avere una minore tendenza all’overfitting e maggiore robustezza a
dati non visti.
- **CADDY**: si tratta di un dataset reale prodotto nel progetto EU FP7 Cognitive autonomus diving buddy in cui vi è un dispositivo di guida autonoma (AUV) che va a collezionare dati ed immagini di sub per task di classificazione oggetti, segmentazione e stima della posa umana. L’ambiente sottomarino ha posto una serie di difficoltà
tecnologiche La prima parte del dataset comprende 10 000 immagini stereo di sub che
interagiscono con l’AUV attraverso i gesti utilizzati per comunicare durante le immersioni. La seconda parte, invece, contiene 12 700 immagini stereo che rappresentano sub
in movimento. Tutte le immagini sono già rettificate, ciò significa che la ricerca di uno
stesso pixel tra il frame di sinistra e quello di destra può essere ristretta considerando
una sola riga nella matrice di pixel che costituiscono le immagini. Sulle 8 scene di CADDY è stato effettuato un partizionamento dei dati, in quanto alcune coppie di immagini
stereo presentavano dei fattori che potevano malcondizionare gli step successivi della
pipeline.

La fase di validazione della rete è stata condotta su dataset non noti in fase di training,
andando quindi a testare la capacità di generalizzazione su dati sconosciuti. Solo stati selezio-
nati tre dataset in particolare per l’analisi dei risultati ottenuti in seguito alla
fase di addestramento.

- **WIRN2022**: è un dataset sottomarino sintetico generato utilizzando software
di modellazione 3D. Comprende 5 scene in cui vi sono roccie di diverse grandezze,
coralli, pneumatici, auto, bottiglie e diverse specie di pesci. Gli oggetti sono posti in
punti casuali nel campo visivo, a distanze diverse dalla camera. In ogni scena sono
rappresentati sia oggetti statici, come roccie, sia elementi dinamici come pesci. Nelle
prime tre scene, la camera è fissata rigidamente e gli elementi sono in moto, mentre
nelle ultime due è presente una combinazione tra il moto della camera e degli elementi.
Ogni immagine ha risoluzione di 960x540 pixels. (credits to dott. V. Scarrica)
- **KITTI2015** : è un dataset reale e dinamico di taglia piccola utilizzato per task di
visione stereo, optical flow, visual odometry, 3D object detection and 3D tracking. Comprende 400 frame suddivisi in uno split 50/50 tra training set e validation set
immagini in contesto urbano. Le immagini sono catturate da una piattaforma di guida
autonoma ed in ogni frame è possibile osservare fino a 15 autoveicoli e diverse decine di
pedoni. La piattaforma vanta un laserscanner 360 gradi Velodyne e una coppia stereo di
camere fissate rigidamente ad una station wagon che percorre aree rurali e superstrade
presso la città di Karlsruhe. Il dataset proposto costituisce una preziosa fonte di dati,
soprattutto per applicazioni di motion estimation e guida autonoma dove, ancora una
volta, sono assenti dati reali con relativo flusso ottico. Le scene sono state interpretate
ed analizzate come una collezione di pochi oggetti rigidi in movimento. E’ quindi lecito
assumere un numero finito molto basso di oggetti rigidi in ogni scena e gli elementi statici,
come lo sfondo, possono essere gestiti separatamente. Per fornire annotazioni dense di
flusso ottico, il primo step è stato quello di aumentare le immagini con ground truth
sparse. Il laserscanner ha permesso di estrarre mappe di disparità e nuvole di punti
che sono state poi combinate geometricamente in software CAD per la generazione
di annotazioni dense. Grazie al setup descritto poc’anzi, le tavole di verità risultano
essere molto accurate, sebbene limitate in numero. La fase di testing condotta su tale
dataset va a testare le capacità del modello di interpretare caratteristiche fotometriche
di immagini reali sconosciute al modello.
- **MPI-SINTEL** : E’ un dataset sintetico, ovvero generato dalla computer graphics,
espressamente per la valutazione di flusso ottico. Deriva da un film animato open-source
di nome “Sintel” prodotto dalla Blender Foundation. 
Ha 23 scene e 1064 immagini stereo con risoluzione 1024x436 per 8bit di profondità. Il
dataset è stato creato con rendering steps a diversa complessità:
  - Albedo pass: le immagini sono renderizzate senza effetti di illuminazione si as-
sume luminosità costante in ogni parte costituente della scena, ad eccezione delle
regioni occluse.
  - Clean pass: le immagini sono renderizzate con la presenza di illuminazione, quindi
vi sono regioni d’ombra e riflessi speculare su superfici come armature e specchi
d’acqua.
  - Final pass: le immagini sono create in full rendering, in aggiunta allo step prece-
dente, sono visibili motion blur, camera depth-of-field blur ed effetti atmosferici.
Ricordando che l’obiettivo è quello di andare a testare in un contesto reale, saran-
no utilizzati solo gli ultimi due step. In tal modo, l’approssimazione risulta essere
maggiormente fedele a livello di dominio e condizioni generali della scena.

## Data ingestion 
Di seguito si presenta brevemente un quadro completo della pipeline in funzione dei dati
utilizzati nell’elaborato. In particolare si possono osservare i dataset presentati, in grigio le diverse strategie di
addestramento e in rosso le unità elaborative, dove Σ corrisponde alla rete di monocular depth
estimation. I tre dataset, sono stati sottoposti alla rete in momenti diversi, secondo
l’ordine numerico mostrato in figura. Gli addestramenti numero 2-3, sono di tipologia
"finetuning" poichè utilizzano la conoscenza appresa durante l’addestramento precedente. La
differenza tra il finetuning di FlyingThings e quello di dCaddy consiste nella configurazione
di iperparametri come numero di iterazioni, learning policy, learning rate.

L’addestramento di RAFT su CADDY è più propriamente indicato con il termine di transfer learning, anche noto come domain adaptation.
Si tratta di un metodo di machine learning in cui si preserva la conoscenza acquisita nel risolvere un problema precedente e la si applica per la risoluzione di un problema simile in un diverso dominio. E’ un approccio molto diffuso nell’addestramento di CNN, poichè è raro avere abbastanza dati da effettuare un addestramento a partire da zero. Pertanto, si usa una configurazione di pesi preaddestrata per inizializzare la rete. Inoltre, architetture molto profonde, sono molto costose in termini computazionali. Possono richiedere anche settimane di calcolo impegnando decine di nodi con GPUs performanti. Nel nostro scenario, andare ad addestrare da zero i 4.8 milioni di parametri di RAFT su un dataset di poche migliaia di immagini andrebbe a penalizzare fortemente l’abilità di generalizzare e correre il rischio
di overfitting. In generale, il transfer learning è applicabile se il dataset non è estremamente diverso dal contesto di pre-addestramento. La parte di feature extraction, ricordiamo, è effettuata a livello di layer convoluzionali 2.4.2. In particolare, le caratteristiche di alto livello come gli angoli e le forme sono individuate nei layer iniziali, mentre le caratteristiche di alto livello vengono catturate dai layer finali. Nell’approccio utilizzato sono state previste tre correzioni principali allo scheduling:
- Last layer freezed: sfruttando l’osservazione precedente, in cui le caratteristiche di base sono estratte nei primi layer; sono stati congelati i pesi ed i bias dei livelli di RAFT, eccetto l’ultimo.
- Learning rate decreased: il learning rate è stato abbassato di un fattore pari a dieci, in modo da aggiornare i pesi dell’ultimo livello senza però cambiarne il significato nella rete. In generale, un learning rate alto può condurre a cattivi risultati a causa degli step nella discesa del gradiente. Ciò riduce certamente i tempi computazionali, ma conduce ad uno stato in cui non si riesce a trovare un punto di minimo globale.
Nell’implementazione proposta è stato utilizzata una strategia di ottimizzazione AdamW con learning rate pari a 0.000125, weight decay=0.0004, betas=(0.9, 0.999).
- Higher number of iterations: dato che il rischio di overfitting è molto basso, il modello è stato processato con un numero massimo di iterazioni tre volte maggiore rispetto a quello utilizzato in fase di pre-addestramento.

## Installation

Please refer to [install.md](docs/en/install.md) for installation and
guidance in [dataset_prepare](docs/en/dataset_prepare.md) for dataset preparation.

## Acknowledgement

MMFlow is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new flow algorithm.

## Citation

```BibTeX
@misc{2021mmflow,
    title={{MMFlow}: OpenMMLab Optical Flow Toolbox and Benchmark},
    author={MMFlow Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmflow}},
    year={2021}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab Model Deployment Framework.
